"""
make_context_chunks.py
======================
Generates token-targeted chunks for each category in contexts.json.
Uses the pointer-based sentence chunking algorithm from Utils.context_chunking,
calibrating against the ground-truth token_count from contexts.json so the
combined sum of chunk tokens matches the original document token count.

The 5 Context Sizes and their chunk compositions:
1. 1k  (1,000 tokens) : Chunks [C0, C1]                     -> 500 + 500
   - Middle Split     : Before [C0] (500), After [C1] (500)
2. 5k  (5,000 tokens) : Chunks [C0, C1, C2, C3]             -> 500 + 500 + 1500 + 2500
   - Middle Split     : Before [C0..C2] (2500), After [C3] (2500)
3. 10k (10,000 tokens): Chunks [C0..C4]                     -> 5000 + 5000
   - Middle Split     : Before [C0..C3] (5000), After [C4] (5000)
4. 25k (25,000 tokens): Chunks [C0..C6]                     -> 12500 + 12500
   - Middle Split     : Before [C0..C5] (12500), After [C6] (12500)
5. 50k (50,000 tokens): Chunks [C0..C7]                     -> 25000 + 25000
   - Middle Split     : Before [C0..C6] (25000), After [C7] (25000)

Calibrated 8-Chunk Target Distribution per Category:
[500, 500, 1500, 2500, 5000, 2500, 12500, 25000] (Sum: 50,000 tokens)
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable imports from Utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.context_chunking import chunk_contexts_file, DEFAULT_TOKEN_TARGETS

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON = BASE_DIR / "contexts.json"
OUTPUT_JSON = BASE_DIR / "contexts_chunked.json"

# Target distribution per category: C0=500, C1=500, C2=1500, C3=2500, C4=5000, C5=2500, C6=12500, C7=25000
TARGET_TOKEN_COUNTS = [500, 500, 1500, 2500, 5000, 2500, 12500, 25000]

# Set to True if you want live Google GenAI API token counts (requires GEMINI_API_KEY)
USE_GEMINI_API = False

# If False, the last chunk collects the remainder (~25,000-26,000 tokens) to preserve 100% of context.
# If True, the last chunk is strictly capped at ~25,000 tokens.
CAP_LAST_CHUNK = True


def main():
    print("=" * 78)
    print("EXPERIMENT A2 - CONTEXT CHUNKING FOR 5 BENCHMARK SIZES")
    print("=" * 78)
    print(f"Project Root   : {PROJECT_ROOT}")
    print(f"Input File     : {INPUT_JSON}")
    print(f"Output File    : {OUTPUT_JSON}")
    print(f"Target Tokens  : {TARGET_TOKEN_COUNTS} (Sum: {sum(TARGET_TOKEN_COUNTS):,})")
    print(f"Cap Last Chunk : {CAP_LAST_CHUNK}")
    print(f"API Counting   : {USE_GEMINI_API}")
    print("-" * 78)
    print("5 CONTEXT SIZES MAPPING:")
    print("  - 1k  ( 1,000 tok) : Chunks [C0, C1]             -> 500 + 500 (mid: 500 / 500)")
    print("  - 5k  ( 5,000 tok) : Chunks [C0..C3]             -> 2500 + 2500 (mid: 2500 / 2500)")
    print("  - 10k (10,000 tok) : Chunks [C0..C4]             -> 5000 + 5000 (mid: 5000 / 5000)")
    print("  - 25k (25,000 tok) : Chunks [C0..C6]             -> 12500 + 12500 (mid: 12500 / 12500)")
    print("  - 50k (50,000 tok) : Chunks [C0..C7]             -> 25000 + 25000 (mid: 25000 / 25000)")
    print("=" * 78)

    if not INPUT_JSON.exists():
        print(f"[Error] Input file not found: {INPUT_JSON}")
        sys.exit(1)

    chunk_contexts_file(
        input_file_path=str(INPUT_JSON),
        output_file_path=str(OUTPUT_JSON),
        target_tokens=TARGET_TOKEN_COUNTS,
        use_gemini_api=USE_GEMINI_API,
        cap_last_chunk=CAP_LAST_CHUNK
    )


if __name__ == "__main__":
    main()

