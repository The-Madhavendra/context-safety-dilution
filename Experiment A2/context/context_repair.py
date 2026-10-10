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
    for parent in Path(__file__).resolve().parents:
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
except ImportError:
    pass

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# The target category to top-up to 50k tokens
TARGET_CATEGORY = "Linguistics & Language Evolution"

# Model options: "google/gemini-3.5-flash-lite", "google/gemini-2.5-flash", "google/gemini-2.0-flash"
MODEL = "google/gemini-3.5-flash-lite"

TARGET_TOTAL_TOKENS = 50000          # Stop condition
MAX_OUTPUT_TOKENS_PER_CALL = 8192    # Max tokens Gemini can produce per call
MAX_REPAIR_CALLS = 10                # Safety ceiling on API calls
SLEEP_SECONDS = 5                    # Rate limit pause

BASE_DIR = Path(__file__).parent
JSON_FILE_PATH = BASE_DIR / "contexts.json"
TXT_FILE_PATH = BASE_DIR / "contexts.txt"


# ==============================================================================
# PROMPT BUILDER
# ==============================================================================
def build_repair_prompt(turn: int, category_name: str, existing_context: str) -> str:
    """
    Feeds existing context to the new chat session and instructs Gemini to
    continue seamlessly with 50,000 words, strictly text-only, no LaTeX.
    """
    return f"""You are generating an exhaustive, encyclopedic, graduate-level reference corpus for the category: {category_name}.

CONTEXT GENERATED SO FAR:
The following continuous text has already been produced for this topic:
\"\"\"
{existing_context}
\"\"\"

CRITICAL CONTINUATION DIRECTIVE:
- Review the text generated so far above. Do NOT repeat or rehash concepts, historical periods, language families, or theoretical debates already covered.
- Pick up immediately from where the text left off and continue writing an exhaustive continuation (generate 50,000 words of comprehensive, deeply detailed content).
- Dive deep into completely new, unaddressed subfields within {category_name} (e.g., historical phonology, computational syntax, comparative philology, paleolinguistics, creole genesis, neurolinguistics, dialectology, endangered language documentation).

STRICT CONTENT & FORMAT REQUIREMENTS:
1. TEXT-ONLY CONTENT: Provide pure, natural prose only.
2. NO LATEX OR MATHEMATICAL EQUATIONS: Do NOT use LaTeX syntax, math blocks ($ or $$), Greek mathematical notation symbols, or equation formatting. If any mathematical formula, physical law, or quantitative relationship exists, it MUST be explained entirely using verbal descriptive words.
3. NO ESCAPE CHARACTERS: Do not use escape characters.
4. CONTINUOUS PROSE: Write clean, dense, highly structured multi-paragraph text. Use standard paragraph breaks.
5. NO PREAMBLE OR CONVERSATIONAL FILLER: Do NOT include greetings (e.g., 'Sure, here is...'), introductory commentary, section numbering, bullet points, Markdown code block fences (do not wrap in ```), or concluding sign-offs.

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

Begin generating the deep continuous continuation text now:
"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def clean_response_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    return text


def update_contexts_json(file_path: Path, category_name: str, new_text: str, token_count: int = None) -> tuple[str, int]:
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

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(contexts_data, f, indent=2, ensure_ascii=False)

    return full_ctx, len(full_ctx)


def save_backup(file_path: Path, raw_text: str, category_name: str, turn: int):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"\n{'='*70}\n"
        f"TIMESTAMP: {timestamp}\n"
        f"CATEGORY : {category_name} (REPAIR SESSION)\n"
        f"CALL     : Turn {turn}\n"
        f"{'='*70}\n"
    )
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(header)
        f.write(raw_text)
        f.write("\n")


# ==============================================================================
# GEMINI CHAT CLIENTS
# ==============================================================================
class GenAIChat:
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
                    wait_time = attempt * 15
                    print(f"[Notice] Model busy / rate limit (503/429). Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                elif attempt < max_retries:
                    wait_time = attempt * 8
                    print(f"[Warning] Request error ({e}). Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise

    def count_tokens(self, text: str) -> tuple[int, bool]:
        try:
            resp = self.client.models.count_tokens(model=self.model_name, contents=text)
            return int(resp.total_tokens), True
        except Exception:
            return int(len(text.split()) * 1.33), False

    def delete(self):
        self.chat = None
        self.client = None


class LegacyGenAIChat:
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
                    print(f"[Notice] Model busy (503/429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt < max_retries:
                    wait_time = attempt * 8
                    print(f"[Warning] Request error ({e}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    def count_tokens(self, text: str) -> tuple[int, bool]:
        try:
            resp = self.model.count_tokens(text)
            return int(resp.total_tokens), True
        except Exception:
            return int(len(text.split()) * 1.33), False

    def delete(self):
        self.chat = None
        self.model = None


class RestApiChat:
    def __init__(self, api_key: str, model_name: str, max_output_tokens: int = 8192):
        self.api_key = api_key
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        self.contents = []

    def send_message(self, text: str, max_retries: int = 6) -> str:
        payload_contents = list(self.contents)
        payload_contents.append({"role": "user", "parts": [{"text": text}]})
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": payload_contents,
            "generationConfig": {"maxOutputTokens": self.max_output_tokens}
        }

        data = None
        for attempt in range(1, max_retries + 1):
            try:
                import requests
                resp = requests.post(self.url, headers=headers, json=payload, timeout=600)
                if resp.status_code in (503, 429):
                    wait_time = attempt * 15
                    print(f"[Notice] Model busy (HTTP {resp.status_code}). Retrying in {wait_time}s...")
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
                    time.sleep(wait_time)
                elif attempt < max_retries:
                    wait_time = attempt * 8
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
        self.contents.append({"role": "user", "parts": [{"text": text}]})
        self.contents.append({"role": "model", "parts": [{"text": reply_text}]})
        return reply_text

    def count_tokens(self, text: str) -> tuple[int, bool]:
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
    clean_model = model_name.split("/")[-1]
    try:
        from google import genai
        return GenAIChat(api_key, clean_model, max_output_tokens)
    except Exception:
        pass

    try:
        import google.generativeai
        return LegacyGenAIChat(api_key, clean_model, max_output_tokens)
    except Exception:
        pass

    return RestApiChat(api_key, clean_model, max_output_tokens)


# ==============================================================================
# MAIN REPAIR ROUTINE
# ==============================================================================
def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Error] No GOOGLE_API_KEY found in environment variables.")
        print("Please set your API key in terminal before running:")
        print("  $env:GOOGLE_API_KEY=\"your_key_here\"")
        sys.exit(1)

    print("=" * 70)
    print(f"=== REPAIR / TOP-UP: {TARGET_CATEGORY} ===")
    print("=" * 70)

    # 1. Read existing context for this category from contexts.json
    if not JSON_FILE_PATH.exists():
        print(f"[Error] {JSON_FILE_PATH} does not exist!")
        sys.exit(1)

    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_obj = None
    for item in data:
        if isinstance(item, dict) and item.get("context_category") == TARGET_CATEGORY:
            target_obj = item
            break

    if not target_obj:
        print(f"[Error] Category '{TARGET_CATEGORY}' not found in {JSON_FILE_PATH}!")
        sys.exit(1)

    existing_context = target_obj.get("context", "")
    if isinstance(existing_context, list):
        existing_context = "\n\n".join(str(x) for x in existing_context)

    # 2. Check current token count from Google API
    checker = create_chat_session(api_key, MODEL)
    initial_tokens, is_real = checker.count_tokens(existing_context)
    checker.delete()
    del checker

    print(f"Existing context length: {len(existing_context):,} chars")
    print(f"Current token count    : {initial_tokens:,} tokens (Target: {TARGET_TOTAL_TOKENS:,})")

    if initial_tokens >= TARGET_TOTAL_TOKENS:
        print(f"[Complete] Category already has {initial_tokens:,} tokens (>= {TARGET_TOTAL_TOKENS:,}). No repair needed!")
        sys.exit(0)

    needed = TARGET_TOTAL_TOKENS - initial_tokens
    print(f"Tokens needed to 50k   : ~{needed:,} tokens\n")

    # 3. Start a brand-new chat session
    chat = create_chat_session(api_key, MODEL, max_output_tokens=MAX_OUTPUT_TOKENS_PER_CALL)
    print(f"[Chat] Fresh chat session created with '{MODEL.split('/')[-1]}'.")

    turn = 0
    cum_tokens = initial_tokens
    current_context = existing_context

    try:
        while cum_tokens < TARGET_TOTAL_TOKENS and turn < MAX_REPAIR_CALLS:
            turn += 1
            print(f"\n{'-'*70}")
            print(f"--- [Repair Call {turn}] Pushing Gemini for continuation ---")
            print(f"Progress: {cum_tokens:,} / {TARGET_TOTAL_TOKENS:,} tokens ({cum_tokens / TARGET_TOTAL_TOKENS * 100:.1f}%) | Need ~{TARGET_TOTAL_TOKENS - cum_tokens:,} more tokens")
            print(f"{'-'*70}")

            prompt = build_repair_prompt(turn, TARGET_CATEGORY, current_context)

            # Send prompt to chat
            raw_response = chat.send_message(prompt)

            # Save raw turn to contexts.txt
            save_backup(TXT_FILE_PATH, raw_response, TARGET_CATEGORY, turn)

            # Clean response text
            addition = clean_response_text(raw_response)
            turn_tokens, is_real = chat.count_tokens(addition)

            # Update context and save to contexts.json
            current_context, char_len = update_contexts_json(JSON_FILE_PATH, TARGET_CATEGORY, addition)

            # Fresh token count on entire concatenated context
            cum_tokens, is_real_cum = chat.count_tokens(current_context)
            src_label = "Google API" if is_real_cum else "estimated"

            # Update the token_count field in contexts.json
            update_contexts_json(JSON_FILE_PATH, TARGET_CATEGORY, "", token_count=cum_tokens)

            print(f"[Yield] Produced: {turn_tokens:,} tokens in this call")
            print(f"[Cumulative] '{TARGET_CATEGORY}' is now {cum_tokens:,} / {TARGET_TOTAL_TOKENS:,} tokens ({src_label}) [{cum_tokens / TARGET_TOTAL_TOKENS * 100:.1f}%]")
            print(f"[Updated] contexts.json updated! Context string is now {char_len:,} chars.")

            if cum_tokens >= TARGET_TOTAL_TOKENS:
                print(f"\n[Goal Reached] Successfully reached {cum_tokens:,} tokens for '{TARGET_CATEGORY}'!")
                break

            print(f"[Wait] Pausing {SLEEP_SECONDS}s for rate limits...")
            time.sleep(SLEEP_SECONDS)

    finally:
        chat.delete()
        del chat
        print(f"\n[Cleanup] Chat session ended and deleted from memory.")

    # 4. Final verification
    print("\n" + "=" * 70)
    print("=== FINAL REPAIR VERIFICATION ===")
    print("=" * 70)
    verifier = create_chat_session(api_key, MODEL)
    final_fresh, _ = verifier.count_tokens(current_context)
    verifier.delete()
    del verifier

    # Sync back
    update_contexts_json(JSON_FILE_PATH, TARGET_CATEGORY, "", token_count=final_fresh)

    print(f"Category            : {TARGET_CATEGORY}")
    print(f"Final Characters    : {len(current_context):,} chars")
    print(f"Final Google Tokens : {final_fresh:,} tokens (Target: {TARGET_TOTAL_TOKENS:,})")
    print(f"Repair Calls Made   : {turn}")
    print(f"File Updated        : {JSON_FILE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
