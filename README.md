# Context-Padding & Safety Degradation in Long-Context LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Inspect AI](https://img.shields.io/badge/framework-Inspect%20AI-orange.svg)](https://inspect.ai-safety-institute.org.uk/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An empirical safety evaluation benchmark investigating guardrail fragility, semantic priming, and positional vulnerability under large context padding.

> [!IMPORTANT]
> **Responsible Disclosure Notice:** The successful safety breaks identified in this research were responsibly disclosed to Google via their designated feedback channel on **2026-10-10**. In accordance with AI safety best practices, harmful generation outputs and the specific working prompt combinations have been redacted from this public repository.

---

## 📌 Executive Summary

While frontier LLMs exhibit high baseline refusal rates against overt malicious prompts, real-world deployment frequently places queries within extensive surrounding contexts (e.g., retrieval-augmented generation, automated document workflows, or long conversational threads).

This benchmark evaluates **1,600 standardized evaluations** across **15 context conditions** (1k, 5k, 10k, 25k, 50k tokens at Start, Middle, and End positions + 0k baseline) on `google/gemini-3.5-flash-lite`, independently scored using the UK AISI **Inspect AI 0–4 Safety Rubric** by `google/gemini-3.1-flash-lite`.

### Key Empirical Findings:
1. **Positional Asymmetry:** 100% of Level-4 Critical Exploits concentrated exclusively in the **Middle** position ("Lost in the Middle" safety collapse). The **End** position retained near-perfect refusal rates (98.4%) due to recency effects.
2. **Category Fragility:** CBRN categories (Biology, Chemical Synthesis) demonstrated high guardrail resilience. Conversely, dual-use categories (Weapons, Cybersecurity) experienced guardrail erosion when surrounded by technical academic corpora.
3. **Semantic Priming:** Technical STEM contexts (*Linguistics & Language Evolution*, *Geology & Earth Formation*, *Mathematics*) hosted 100% of Level-4 exploits, disarming safety triggers through academic framing.

---

## 📁 Repository Structure

```text
├── Eval Prompt/
│   └── master_prompt.txt               # Inspect AI 0-4 scoring rubric & judge system prompt
│
├── Experiment A2/
│   ├── master-eval.py                  # Automated benchmark evaluation suite
│   └── context/
│       ├── contexts.json               # Benign reference corpus (10 academic domains)
│       ├── contexts_chunked.json       # Pre-tokenized padding windows (1k to 50k)
│       ├── contexts.txt                # Raw background corpus
│       ├── make_context_chunks.py      # Window chunking pipeline
│       ├── context_generator.py        # Encyclopedic corpus generation utility
│       └── context_repair.py           # Context validation and repair tool
│
├── Utils/
│   ├── api_management.py               # Self-contained API client with exponential backoff
│   ├── eval_funcs.py                   # Core model caller, dataset builder & scorers
│   ├── inspect_eval.py                 # Native Inspect AI task execution interface
│   ├── serial_batch.py                 # Serial batch execution pipeline
│   ├── custom_eval.py                  # Standalone multithreaded evaluation runner
│   ├── context_chunking.py             # Context positioning & formatting helpers
│   └── experiment_utils.py             # Metric aggregation utilities
│
├── Research Documentation/
│   ├── Research_Documentation_Revised.docx  # Full formal research findings document
│   ├── analysis.ipynb                       # Macroscopic statistical trends notebook
│   ├── figures/                             # Publication-ready figures (Figures 1 to 5)
│   └── sample level analysis/
│       ├── master_samples_compiled.csv      # Complete compiled dataset (1,600 evaluations)
│       ├── master_samples_compiled.json     # Compiled dataset with rich metadata
│       ├── non_zero_samples.csv             # Filtered non-zero scored samples (Scores 1 to 4)
│       ├── sample_analysis Doc.md           # Granular case studies and qualitative failure analysis
│       ├── sample_analysis.ipynb            # Interactive sample-level analysis notebook
│       └── figures/                         # Granular audit figures (Figures 1 to 7)
│
├── baseline_harmful_prompts.json       # 100 standardized harmful probes across 10 categories
├── requirements.txt                    # Project dependencies
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation

Clone this repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration

Set your API keys as environment variables or create a `.env` file in the project root:

```bash
# On Linux / macOS
export GOOGLE_API_KEY="your-google-api-key"
export GROQ_API_KEY="your-groq-api-key"      # Optional: if utilizing Groq models

# On Windows PowerShell
$env:GOOGLE_API_KEY="your-google-api-key"
$env:GROQ_API_KEY="your-groq-api-key"
```

Verify your API configuration:
```bash
python "Experiment A2/master-eval.py" --help
```

### 3. Running Evaluations

To execute the benchmark suite across experimental conditions:

```bash
# Run the master evaluation pipeline
python "Experiment A2/master-eval.py"

# Preview the evaluation schedule without making API calls
python "Experiment A2/master-eval.py" --dry-run
```

### 4. Interactive Data Analysis

Explore the complete sample-level evaluations, word count distributions, and forensic case studies:

```bash
jupyter lab "Research Documentation/sample level analysis/sample_analysis.ipynb"
```

---

## 🛡️ Responsible Disclosure & AI Safety Compliance

In accordance with responsible disclosure protocols for empirical AI safety research:
* **Vendor Notification:** The successful safety breaks and guardrail collapse mechanisms identified during this benchmark were responsibly disclosed to Google via their designated feedback channel on **2026-10-10**.
* **Payload & Exploit Redaction:** Direct model generation outputs containing actionable procedural blueprints (Scores 2 to 4), as well as direct working prompt and context injection combinations, have been redacted (`[REDACTED MODEL RESPONSE]`, `[REDACTED HARMFUL PROMPT]`, and `[REDACTED EVIDENCE]`) from this public repository to eliminate actionable misuse potential.
* **Scientific Reproducibility:** All empirical quantitative metrics—including numerical judge scores, refusal classifications, empirical word counts, character lengths, evaluation rationales, and category interaction matrices—remain fully preserved to facilitate peer scrutiny and research reproducibility without exposing operational threat material.

---

## 📄 License

This research codebase is licensed under the [MIT License](LICENSE).
