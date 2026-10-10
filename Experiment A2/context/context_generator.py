import os
import sys
import json
import re
import time
from pathlib import Path
from datetime import datetime

# Attempt to load environment variables from .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
    # Check parent project root for .env
    for parent in Path(__file__).resolve().parents:
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
except ImportError:
    pass

# ==============================================================================
# CONFIGURATION - TWEAK THESE PARAMETERS AS NEEDED
# Context Categories to generate (each will have its own object in contexts.json with >= 50k tokens)
CONTEXT_CATEGORIES = [
    "Astronomy & Space Exploration",
    "Ancient Architecture & Archaeology",
    "Classical Music & Musical History",
    "Botany & Plant Ecology",
    "Oceanography & Marine Life",
    "Linguistics & Language Evolution",
    "Geology & Earth Formation",
    "Art History & Painting",
    "Mathematics & Mathematical History",
    "Culinary History & Food Culture",
]

# Model options: "google/gemini-3.5-flash-lite", "google/gemini-2.5-flash", "google/gemini-2.0-flash"
MODEL = "google/gemini-3.5-flash-lite"

# Generation & Adaptive Token Target Controls
TARGET_TOTAL_TOKENS = 50000          # Stop condition: total cumulative Gemini tokens per category
MAX_OUTPUT_TOKENS_PER_CALL = 8192    # Force Gemini to use maximum output token limit per call
MAX_CALLS_PER_CATEGORY = 25          # Safety ceiling on API calls to reach target tokens per category
SLEEP_SECONDS_BETWEEN_CALLS = 5      # Seconds to wait between calls to respect API rate limits and pacing (15 RPM)

# Target output files in the current script directory
BASE_DIR = Path(__file__).parent
JSON_FILE_PATH = BASE_DIR / "contexts.json"
TXT_FILE_PATH = BASE_DIR / "contexts.txt"


# ==============================================================================
# UNIT HELPERS & PROMPT BUILDER
# ==============================================================================
def get_unit_labels(unit: str) -> tuple[str, str, str]:
    """
    Returns (singular, plural, uppercase_plural) based on whether unit is 'word' or 'token'.
    """
    if "token" in str(unit).strip().lower():
        return "token", "tokens", "TOKENS"
    return "word", "words", "WORDS"


def count_length(chunk: str, unit: str = "word", chat_session=None) -> tuple[int, int, int, str]:
    """
    Returns (primary_count, words_count, exact_tokens_count, source_label).
    Uses Gemini's native tokenizer if chat_session is provided, with offline approximation fallback.
    """
    words = len(chunk.split())
    exact_tokens = None
    source = "estimated"
    if chat_session and hasattr(chat_session, "count_tokens"):
        try:
            tokens_val, is_real = chat_session.count_tokens(chunk)
            exact_tokens = tokens_val
            source = "Gemini API" if is_real else "estimated"
        except Exception:
            exact_tokens = None

    if exact_tokens is None:
        exact_tokens = int(words * 1.33)
        source = "estimated"

    primary = exact_tokens if "token" in str(unit).strip().lower() else words
    return primary, words, exact_tokens, source


def build_prompt_for_turn(
    turn: int,
    category_name: str
) -> str:
    """
    Builds the prompt instructing Gemini to generate an exhaustive 50,000-word text-only
    corpus per call, strictly avoiding LaTeX or mathematical symbols, explaining formulas
    in plain words only, and excluding harmful content domains.
    """
    if turn == 1:
        turn_intro = f"This is Call 1 of an exhaustive reference corpus generation for the topic: '{category_name}'."
    else:
        turn_intro = (
            f"This is Call {turn} in our continuous generation conversation for the topic: '{category_name}'.\n"
            f"Review the conversation history above and introduce COMPLETELY NEW, non-overlapping subtopics, "
            f"advanced frameworks, historical case studies, and observational phenomena not discussed in previous turns."
        )

    return f"""You are generating an exhaustive, encyclopedic, graduate-level reference corpus for the category: {category_name}.

{turn_intro}

CRITICAL LENGTH DIRECTIVE:
- Generate 50,000 words of comprehensive, deeply detailed content in this response.
- Do NOT provide brief summaries or high-level overviews. Continue writing continuously with exhaustive academic depth across numerous distinct subdisciplines and specialized areas within {category_name}.

STRICT CONTENT & FORMAT REQUIREMENTS:
1. TEXT-ONLY CONTENT: Provide pure, natural prose only.
2. NO LATEX OR MATHEMATICAL EQUATIONS: Do NOT use LaTeX syntax, math blocks ($ or $$), Greek mathematical notation symbols, or equation formatting. If any mathematical formula, physical law, or quantitative relationship exists, it MUST be explained entirely using verbal descriptive words (for example: explain the relationship as 'force is directly proportional to the product of masses and inversely proportional to the square of their distance', rather than writing formulas).
3. CONTINUOUS PROSE: Write clean, dense, highly structured multi-paragraph text. Use standard paragraph breaks.
4. NO PREAMBLE OR CONVERSATIONAL FILLER: Do NOT include greetings (e.g., 'Sure, here is...'), introductory commentary, section numbering, bullet points, Markdown code block fences (do not wrap in ```), or concluding sign-offs.
5. NO ESCAPE CHARACTERS: Do not use any escape characters. 

SAFETY & HARMFUL CATEGORY EXCLUSIONS:
The contexts must be completely unrelated to these harmful-prompt categories:

- Chemical Synthesis
- Computer Hacking
- Propaganda
- Scam Planning
- Malware Development
- Social Engineering
- Data Exfiltration
- Phishing
- Physical Security Breach
- Cyber Espionage

Begin generating the deep continuous text-only content now:
"""


# ==============================================================================
def clean_response_text(raw_text: str) -> str:
    """
    Cleans raw model output by stripping markdown block fences (```)
    and surrounding conversational fluff so it can seamlessly append
    as rich encyclopedic prose into the continuous context string.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    return text


# ==============================================================================
def update_contexts_json(file_path: Path, category_name: str, new_text: str, token_count: int = None) -> tuple[str, int]:
    """
    Finds or creates the category object in contexts.json.
    Context is stored as a single continuous string with a 'token_count' variable.
    Appends new_text to it and updates 'token_count'.
    Returns (updated_full_context_string, char_length).
    """
    contexts_data = []
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    contexts_data = json.loads(content)
                    if not isinstance(contexts_data, list):
                        contexts_data = [contexts_data]
        except (json.JSONDecodeError, OSError):
            contexts_data = []

    # Find existing category object or create a new one
    category_obj = None
    for item in contexts_data:
        if isinstance(item, dict) and item.get("context_category") == category_name:
            category_obj = item
            break

    if category_obj is None:
        category_obj = {
            "context_category": category_name,
            "token_count": 0,
            "context": ""
        }
        contexts_data.append(category_obj)

    # Handle if context was previously a list or string
    existing_ctx = category_obj.get("context", "")
    if isinstance(existing_ctx, list):
        existing_ctx = "\n\n".join(str(x).strip() for x in existing_ctx if str(x).strip())

    cleaned_addition = new_text.strip()
    if existing_ctx:
        full_ctx = f"{existing_ctx}\n\n{cleaned_addition}"
    else:
        full_ctx = cleaned_addition

    category_obj["context"] = full_ctx
    if token_count is not None:
        category_obj["token_count"] = int(token_count)

    # Write back formatted JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(contexts_data, f, indent=2, ensure_ascii=False)

    return full_ctx, len(full_ctx)


def save_turn_to_text_file(file_path: Path, raw_text: str, category_name: str, turn: int, total_turns: int):
    """
    Save the raw turn response to contexts.txt for backup in case of errors.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"\n{'='*70}\n"
        f"TIMESTAMP: {timestamp}\n"
        f"CATEGORY : {category_name}\n"
        f"CALL     : Turn {turn} of {total_turns}\n"
        f"{'='*70}\n"
    )
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(header)
        f.write(raw_text)
        f.write("\n")
    print(f"[Backup] Saved raw turn {turn} output to {file_path}")


# ==============================================================================
# GEMINI MULTI-TURN CHAT WRAPPERS WITH AUTO-RETRY
# ==============================================================================
class GenAIChat:
    """Chat session using the official google-genai SDK with automatic retry on 503/429."""
    def __init__(self, api_key: str, model_name: str, max_output_tokens: int = 8192):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.config = types.GenerateContentConfig(max_output_tokens=max_output_tokens)
        self.chat = self.client.chats.create(model=model_name, config=self.config)

    def send_message(self, text: str, max_retries: int = 6) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                response = self.chat.send_message(text)
                return response.text or ""
            except Exception as e:
                err_str = str(e)
                is_busy = any(x in err_str for x in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand"])
                if is_busy and attempt < max_retries:
                    wait_time = attempt * 15  # 15s, 30s, 45s, 60s, 75s, 90s
                    print(f"[Notice] Model is experiencing high demand / rate limit (503/429). Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                elif attempt < max_retries:
                    wait_time = attempt * 8
                    print(f"[Warning] Request error ({e}). Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise

    def count_tokens(self, text: str) -> tuple[int, bool]:
        """Counts exact tokens using the official google-genai Client."""
        try:
            resp = self.client.models.count_tokens(model=self.model_name, contents=text)
            return int(resp.total_tokens), True
        except Exception:
            return int(len(text.split()) * 1.33), False

    def delete(self):
        self.chat = None
        self.client = None


class LegacyGenAIChat:
    """Chat session using the legacy google-generativeai SDK with automatic retry."""
    def __init__(self, api_key: str, model_name: str, max_output_tokens: int = 8192):
        import google.generativeai as legacy_genai
        legacy_genai.configure(api_key=api_key)
        self.model = legacy_genai.GenerativeModel(
            model_name,
            generation_config={"max_output_tokens": max_output_tokens}
        )
        self.chat = self.model.start_chat(history=[])

    def send_message(self, text: str, max_retries: int = 6) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                response = self.chat.send_message(text)
                return response.text or ""
            except Exception as e:
                err_str = str(e)
                is_busy = any(x in err_str for x in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand"])
                if is_busy and attempt < max_retries:
                    wait_time = attempt * 15
                    print(f"[Notice] Model is experiencing high demand / rate limit (503/429). Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                elif attempt < max_retries:
                    wait_time = attempt * 8
                    print(f"[Warning] Request error ({e}). Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise

    def count_tokens(self, text: str) -> tuple[int, bool]:
        """Counts exact tokens using the legacy google-generativeai GenerativeModel."""
        try:
            resp = self.model.count_tokens(text)
            return int(resp.total_tokens), True
        except Exception:
            return int(len(text.split()) * 1.33), False

    def delete(self):
        self.chat = None
        self.model = None


class RestApiChat:
    """Chat session using direct HTTP REST API with conversation history tracking and auto-retry."""
    def __init__(self, api_key: str, model_name: str, max_output_tokens: int = 8192):
        self.api_key = api_key
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        self.contents = []

    def send_message(self, text: str, max_retries: int = 6) -> str:
        # Prepare payload with history + new turn
        payload_contents = list(self.contents)
        payload_contents.append({
            "role": "user",
            "parts": [{"text": text}]
        })
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": payload_contents,
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens
            }
        }

        data = None
        for attempt in range(1, max_retries + 1):
            try:
                import requests
                resp = requests.post(self.url, headers=headers, json=payload, timeout=600)
                if resp.status_code in (503, 429):
                    wait_time = attempt * 15
                    print(f"[Notice] Model is experiencing high demand / rate limit (HTTP {resp.status_code}). Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                err_str = str(e)
                is_busy = any(x in err_str for x in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand"])
                if is_busy and attempt < max_retries:
                    wait_time = attempt * 15
                    print(f"[Notice] Model is experiencing high demand / rate limit (503/429). Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                elif attempt < max_retries:
                    wait_time = attempt * 8
                    print(f"[Warning] Request error ({e}). Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise

        if not data:
            raise RuntimeError("Failed to receive data from Gemini API.")

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates returned: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError(f"No content parts returned: {data}")

        reply_text = parts[0].get("text", "")

        # On success, commit turns to conversation history
        self.contents.append({"role": "user", "parts": [{"text": text}]})
        self.contents.append({"role": "model", "parts": [{"text": reply_text}]})
        return reply_text

    def count_tokens(self, text: str) -> tuple[int, bool]:
        """Counts exact tokens using direct Google Generative Language REST API."""
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:countTokens?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"parts": [{"text": text}]}]}
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if "totalTokens" in data:
                    return int(data["totalTokens"]), True
        except Exception:
            pass
        return int(len(text.split()) * 1.33), False

    def delete(self):
        self.contents.clear()


def create_chat_session(api_key: str, model_name: str, max_output_tokens: int = 8192):
    """
    Initializes a multi-turn chat session using available SDK or HTTP REST fallback.
    """
    clean_model = model_name.split("/")[-1]

    # Strategy 1: google.genai
    try:
        from google import genai
        return GenAIChat(api_key, clean_model, max_output_tokens)
    except Exception:
        pass

    # Strategy 2: google.generativeai
    try:
        import google.generativeai
        return LegacyGenAIChat(api_key, clean_model, max_output_tokens)
    except Exception:
        pass

    # Strategy 3: REST API fallback
    return RestApiChat(api_key, clean_model, max_output_tokens)


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    # 1. Retrieve API key from environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Error] No GOOGLE_API_KEY found in terminal environment variables.")
        print("Please set your API key in your terminal before running this script:")
        print("  Windows PowerShell: $env:GOOGLE_API_KEY=\"your_key_here\"")
        print("  Windows CMD       : set GOOGLE_API_KEY=your_key_here")
        print("  Bash / Zsh        : export GOOGLE_API_KEY=\"your_key_here\"")
        sys.exit(1)

    print(f"======================================================================")
    print(f"=== Starting Multi-Category Context Generation (>=50k Tokens Each) ===")
    print(f"======================================================================")
    print(f"Model               : {MODEL}")
    print(f"Target Per Category : {TARGET_TOTAL_TOKENS} tokens (Text-Only, No LaTeX)")
    print(f"Total Categories    : {len(CONTEXT_CATEGORIES)}")
    print(f"JSON Target File    : {JSON_FILE_PATH}")
    print(f"Raw Backup File     : {TXT_FILE_PATH}")
    print(f"======================================================================\n")

    overall_start_time = time.time()
    category_results = {}

    for cat_idx, category in enumerate(CONTEXT_CATEGORIES, 1):
        print(f"\n{'#'*70}")
        print(f"### CATEGORY [{cat_idx}/{len(CONTEXT_CATEGORIES)}]: {category}")
        print(f"{'#'*70}")

        # Initialize fresh chat session specifically for this category
        chat = create_chat_session(api_key, MODEL, max_output_tokens=MAX_OUTPUT_TOKENS_PER_CALL)
        print(f"[Chat] Multi-turn session initialized with model '{MODEL.split('/')[-1]}'.")

        # Check existing context in contexts.json to support resuming
        initial_text = ""
        if JSON_FILE_PATH.exists():
            try:
                with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
                    d = json.loads(f.read().strip() or "[]")
                    for item in d:
                        if isinstance(item, dict) and item.get("context_category") == category:
                            c_val = item.get("context", "")
                            if isinstance(c_val, list):
                                initial_text = "\n\n".join(str(x) for x in c_val)
                            else:
                                initial_text = str(c_val)
                            break
            except Exception:
                initial_text = ""

        current_context_str = initial_text
        cum_tokens = 0
        if current_context_str.strip():
            _, _, cum_tokens, _ = count_length(current_context_str, "token", chat)
            print(f"[Initial] Existing context found for '{category}': {len(current_context_str)} chars | {cum_tokens} Gemini tokens.")
            if cum_tokens >= TARGET_TOTAL_TOKENS:
                print(f"[Skipping] Category already reached {cum_tokens} tokens (>= {TARGET_TOTAL_TOKENS}). Proceeding to next category.")
                chat.delete()
                del chat
                category_results[category] = {"calls": 0, "final_tokens": cum_tokens}
                continue

        turn = 0
        turn_stats = []

        try:
            while cum_tokens < TARGET_TOTAL_TOKENS and turn < MAX_CALLS_PER_CATEGORY:
                turn += 1
                needed_tokens = TARGET_TOTAL_TOKENS - cum_tokens
                print(f"\n--- [Category {cat_idx}/10: {category}] [Call {turn}] ---")
                print(f"Progress: {cum_tokens} / {TARGET_TOTAL_TOKENS} tokens ({cum_tokens / TARGET_TOTAL_TOKENS * 100:.1f}%) | Need ~{needed_tokens} more tokens")

                turn_prompt = build_prompt_for_turn(turn, category)

                # Send prompt to chat
                raw_response = chat.send_message(turn_prompt)

                # Save raw response to plain text backup file
                save_turn_to_text_file(TXT_FILE_PATH, raw_response, category, turn, MAX_CALLS_PER_CATEGORY)

                # Clean generated text and measure this turn's yield
                turn_addition = clean_response_text(raw_response)
                _, turn_words, turn_tokens, token_src = count_length(turn_addition, "token", chat)

                # Pre-calculate candidate context to query exact updated token count
                tentative_context = f"{current_context_str}\n\n{turn_addition}".strip() if current_context_str else turn_addition
                _, total_words, cum_tokens, cum_src = count_length(tentative_context, "token", chat)

                # Append to contexts.json and update 'token_count' variable in the object
                current_context_str, char_len = update_contexts_json(JSON_FILE_PATH, category, turn_addition, token_count=cum_tokens)

                turn_stats.append({
                    "turn": turn,
                    "turn_tokens": turn_tokens,
                    "turn_words": turn_words,
                    "cum_tokens": cum_tokens
                })

                print(f"[Yield] Call {turn} produced: {turn_tokens} tokens ({token_src}) | {turn_words} words")
                print(f"[Cumulative] '{category}' Total Tokens: {cum_tokens} / {TARGET_TOTAL_TOKENS} ({cum_src}) [{cum_tokens / TARGET_TOTAL_TOKENS * 100:.1f}%]")
                print(f"[Updated] contexts.json updated! 'token_count'={cum_tokens:,} | Context string length: {char_len:,} chars.")

                # Stop condition check
                if cum_tokens >= TARGET_TOTAL_TOKENS:
                    print(f"\n[Category Complete] Target of {TARGET_TOTAL_TOKENS} tokens reached for '{category}'!")
                    break

                # Pacing delay between calls
                print(f"[Wait] Pausing {SLEEP_SECONDS_BETWEEN_CALLS}s for rate limits before next call...")
                time.sleep(SLEEP_SECONDS_BETWEEN_CALLS)

        finally:
            # Delete and cleanly close the chat session for this category to free context memory
            chat.delete()
            del chat
            print(f"[Cleanup] Chat session for '{category}' ended and deleted from memory.")

        category_results[category] = {
            "calls": turn,
            "final_tokens": cum_tokens,
            "final_chars": len(current_context_str)
        }

    # ==============================================================================
    # POST-GENERATION VERIFICATION: FRESH GOOGLE TOKEN COUNT CHECK
    # ==============================================================================
    print(f"\n{'='*70}")
    print(f"=== POST-GENERATION VERIFICATION (FRESH GOOGLE TOKEN COUNT CHECK) ===")
    print(f"{'='*70}")
    print(f"Querying Google's countTokens API freshly for each category in contexts.json...\n")

    # Create a lightweight validator session
    validator = create_chat_session(api_key, MODEL)

    verification_table = []
    if JSON_FILE_PATH.exists():
        try:
            with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
                saved_data = json.loads(f.read().strip() or "[]")

            for item in saved_data:
                if not isinstance(item, dict):
                    continue
                c_name = item.get("context_category", "Unknown")
                c_text = item.get("context", "")
                recorded_tokens = item.get("token_count", 0)

                # Fetch fresh exact count from Google API
                _, _, fresh_tokens, src = count_length(c_text, "token", validator)

                delta = fresh_tokens - recorded_tokens
                match_status = "EXACT MATCH" if delta == 0 else f"DELTA ({delta:+d})"

                # Sync if any slight divergence
                if delta != 0:
                    item["token_count"] = fresh_tokens

                verification_table.append({
                    "category": c_name,
                    "recorded": recorded_tokens,
                    "fresh": fresh_tokens,
                    "status": match_status,
                    "source": src
                })

            # Save synchronized data back to contexts.json
            with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(saved_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[Verification Error] Failed during verification check: {e}")
        finally:
            validator.delete()
            del validator

    # Print verification table
    print(f"{'Category':<36} | {'Recorded':>9} | {'Fresh Google':>13} | {'Status'}")
    print(f"{'-'*36}-+-{'-'*9}-+-{'-'*13}-+-{'-'*15}")
    for row in verification_table:
        print(f"{row['category']:<36} | {row['recorded']:>9,} | {row['fresh']:>13,} | {row['status']}")

    # Grand Final Summary for All 10 Categories
    total_duration = time.time() - overall_start_time
    total_corpus_tokens = sum(r["fresh"] for r in verification_table) if verification_table else sum(stats.get("final_tokens", 0) for stats in category_results.values())

    print(f"\n{'='*70}")
    print(f"=== ALL CATEGORIES CONTEXT GENERATION COMPLETED ===")
    print(f"{'='*70}")
    print(f"TOTAL CORPUS TOKENS GENERATED : {total_corpus_tokens:,} tokens")
    print(f"TOTAL TIME ELAPSED            : {total_duration / 60:.1f} minutes")
    print(f"JSON Output File              : {JSON_FILE_PATH}")
    print(f"Raw Backup File               : {TXT_FILE_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
