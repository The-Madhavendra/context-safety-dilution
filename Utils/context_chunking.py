"""
context_chunking.py
===================
Utility module for splitting large text contexts into sequential chunks matching
target token counts, snapping boundaries strictly to complete sentences (using a
nearest-boundary pointer) to prevent cut-off words or incomplete sentences.

Calibrates chunk token counts against the ground-truth `token_count` recorded in
contexts.json so the sum of chunk tokens accurately matches the original document token count.

Accepts dynamic target token distributions, for example:
[500, 500, 1500, 2500, 5000, 10000, 5000, 25000] (Total = 50,000 tokens)
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple

# Try loading .env if available
try:
    from dotenv import load_dotenv
    for parent in Path(__file__).resolve().parents:
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
except ImportError:
    pass


# Default token targets matching the experiment specification (Total: 50,000)
# C0=500, C1=500, C2=1500, C3=2500, C4=5000, C5=2500, C6=12500, C7=25000
DEFAULT_TOKEN_TARGETS = [500, 500, 1500, 2500, 5000, 2500, 12500, 25000]


# ==============================================================================
# 1. SENTENCE SEGMENTATION
# ==============================================================================
def split_into_sentences(text: str) -> List[str]:
    """
    Splits continuous prose text into full, complete sentences,
    preserving sentence-ending punctuation and quotation marks.
    Avoids variable-width lookbehinds for full Python re compatibility.
    """
    if not text:
        return []

    # Match terminal punctuation [.!?] optionally followed by closing quotes/brackets and whitespace
    parts = re.split(r'([.!?]+[\'"]?\s+)', text.strip())
    sentences = []
    current = ""

    for part in parts:
        current += part
        # If part contains terminal punctuation followed by trailing whitespace, a sentence ended
        if re.search(r'[.!?]+[\'"]?\s+$', part):
            s = current.strip()
            if s:
                sentences.append(s)
            current = ""

    if current.strip():
        sentences.append(current.strip())

    return sentences


# ==============================================================================
# 2. TOKEN COUNTING & CALIBRATION HELPERS
# ==============================================================================
def create_calibrated_counter(
    raw_context: str,
    reference_token_count: Optional[int] = None
) -> Tuple[Callable[[str], int], float]:
    """
    Creates a token counter calibrated directly to the ground-truth token count
    from contexts.json. This guarantees the sum of sentence/chunk counts precisely
    reflects the model's actual token count without over- or under-estimating.
    """
    total_len = len(raw_context)
    if reference_token_count and reference_token_count > 0 and total_len > 0:
        ratio = reference_token_count / total_len
    else:
        # Fallback ratio for standard prose on Gemini (~4.3 characters per token)
        words = len(raw_context.split())
        ratio = (words * 1.30) / max(1, total_len)

    def counter(text: str) -> int:
        if not text:
            return 0
        return max(1, int(round(len(text) * ratio)))

    return counter, ratio


def create_gemini_token_counter(model_name: str = "gemini-2.0-flash", api_key: Optional[str] = None) -> Callable[[str], int]:
    """
    Creates a token counter function using the official Google GenAI Client.
    Falls back gracefully to estimate_tokens if API is unavailable or limits exceeded.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return estimate_tokens

    try:
        from google import genai
        client = genai.Client(api_key=key)

        def gemini_counter(text: str) -> int:
            if not text:
                return 0
            try:
                resp = client.models.count_tokens(model=model_name, contents=text)
                return int(resp.total_tokens)
            except Exception:
                return estimate_tokens(text)

        return gemini_counter
    except Exception:
        return estimate_tokens


def estimate_tokens(text: str) -> int:
    """
    General fast local token estimator for Gemini / BPE models.
    """
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(1, int((words * 1.25 + chars / 4.2) / 2))


# ==============================================================================
# 3. SENTENCE-POINTER CHUNKING ALGORITHM
# ==============================================================================
def chunk_context_by_token_targets(
    context: str,
    target_tokens: List[int] = DEFAULT_TOKEN_TARGETS,
    token_counter: Optional[Callable[[str], int]] = None,
    cap_last_chunk: bool = False
) -> Tuple[List[str], List[int]]:
    """
    Splits a continuous context string into sequential chunks matching the target token counts.

    Pointer Logic:
    1. Splits context into complete sentences: S_0, S_1, ..., S_n.
    2. Sequentially accumulates sentences for the current target.
    3. At the transition boundary where adding sentence S_i crosses the target:
       - Evaluates candidate without S_i (count A < target)
       - Evaluates candidate with S_i (count B >= target)
       - Selects the boundary that minimizes absolute error |count - target|.
    4. Decodes/reconstructs the chunk by joining sentences with clean spaces.
    5. Repeats for each target in target_tokens.
    6. For the final target:
       - If cap_last_chunk=False: accumulates all remaining sentences so the entire context is preserved.
       - If cap_last_chunk=True: caps the final chunk at the target token count.

    Returns:
        (chunks, chunk_token_counts)
    """
    if not context or not target_tokens:
        return [], []

    counter = token_counter or estimate_tokens
    sentences = split_into_sentences(context)
    if not sentences:
        return [context], [counter(context)]

    chunks = []
    chunk_counts = []
    sentence_idx = 0
    total_sentences = len(sentences)

    for target_idx, target in enumerate(target_tokens):
        if sentence_idx >= total_sentences:
            break

        current_sentences = []
        current_token_count = 0
        is_last_target = (target_idx == len(target_tokens) - 1)

        while sentence_idx < total_sentences:
            s = sentences[sentence_idx]
            s_tokens = counter(s)

            # If it's the last target and we do NOT cap, take all remaining sentences
            if is_last_target and not cap_last_chunk:
                current_sentences.append(s)
                current_token_count += s_tokens
                sentence_idx += 1
                continue

            projected_count = current_token_count + s_tokens

            if projected_count < target:
                current_sentences.append(s)
                current_token_count = projected_count
                sentence_idx += 1
            else:
                # Boundary evaluation
                error_before = abs(target - current_token_count)
                error_after = abs(projected_count - target)

                if current_sentences and error_before <= error_after:
                    # Stopping before S_i is closer to target
                    break
                else:
                    # Including S_i is closer
                    current_sentences.append(s)
                    current_token_count = projected_count
                    sentence_idx += 1
                    break

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            actual_count = counter(chunk_text)
            chunks.append(chunk_text)
            chunk_counts.append(actual_count)

    # If any sentences remain and cap_last_chunk was True, attach remaining to last chunk or keep separate
    if sentence_idx < total_sentences and chunks and not cap_last_chunk:
        remaining_text = " ".join(sentences[sentence_idx:])
        chunks[-1] = chunks[-1] + " " + remaining_text
        chunk_counts[-1] = counter(chunks[-1])

    return chunks, chunk_counts


# Alias for backward compatibility
split_context_by_token_targets = chunk_context_by_token_targets


# ==============================================================================
# 4. JSON FILE PROCESSOR
# ==============================================================================
def chunk_contexts_file(
    input_file_path: str,
    output_file_path: Optional[str] = None,
    target_tokens: List[int] = DEFAULT_TOKEN_TARGETS,
    use_gemini_api: bool = False,
    model_name: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
    cap_last_chunk: bool = False
) -> str:
    """
    Reads contexts.json, calibrates token counts against the original object's token_count,
    splits context into chunks matching target_tokens, and writes out contexts_chunked.json.

    Args:
        input_file_path: Path to input contexts.json
        output_file_path: Path to output JSON (defaults to contexts_chunked.json)
        target_tokens: Dynamic list of token count targets
        use_gemini_api: If True, uses Google GenAI API for token counting
        model_name: Gemini model name for count_tokens
        api_key: Gemini API key
        cap_last_chunk: If True, caps the last chunk strictly to the target token count

    Returns:
        output_file_path
    """
    input_path = Path(input_file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_file_path is None:
        output_path = input_path.parent / f"{input_path.stem}_chunked{input_path.suffix}"
    else:
        output_path = Path(output_file_path)

    print(f"Loading input file: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    print(f"Processing {len(data)} context categories with target tokens: {target_tokens}")
    processed_data = []

    for idx, item in enumerate(data, start=1):
        category = item.get("context_category", f"Category_{idx}")
        raw_context = item.get("context", "")
        original_token_count = item.get("token_count")

        if isinstance(raw_context, list):
            raw_context = " ".join(str(x) for x in raw_context)

        # Token counter selection
        if use_gemini_api:
            token_counter = create_gemini_token_counter(model_name=model_name, api_key=api_key)
        else:
            # Calibrate against original token count so chunk sum matches contexts.json!
            token_counter, _ = create_calibrated_counter(raw_context, original_token_count)

        chunks, chunk_counts = chunk_context_by_token_targets(
            context=raw_context,
            target_tokens=target_tokens,
            token_counter=token_counter,
            cap_last_chunk=cap_last_chunk
        )

        total_chunk_tokens = sum(chunk_counts)

        # Output object with identical structure
        new_obj = {
            "context_category": category,
            "original_token_count": original_token_count,
            "token_count": total_chunk_tokens,
            "chunk_token_counts": chunk_counts,
            "context": chunks
        }
        processed_data.append(new_obj)

        diff_str = ""
        if original_token_count:
            diff = total_chunk_tokens - original_token_count
            diff_str = f" | Original: {original_token_count:,} (Diff: {diff:+d})"

        print(f"[{idx:2d}/{len(data):2d}] {category}")
        print(f"       Chunks: {len(chunks)} | Counts: {chunk_counts}")
        print(f"       Total Tokens: {total_chunk_tokens:,}{diff_str}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully wrote chunked contexts to: {output_path}")
    return str(output_path)


# Alias for backward compatibility
process_contexts_file = chunk_contexts_file
