# chuangzhi-summer-camp · Harness Engineering

> **SII 2026 Summer Camp · Harness Engineering Submission**
> Under hard constraints of **2048 prompt-token budget**, **frozen Qwen3-8B weights**, and **stdlib + numpy only**, build a retrieval-based few-shot harness that handles a mixed workload of customer-intent classification, OOD classification, and A/B/C/D multiple-choice questions.

📄 [**Exploration Report (PDF)**](探索报告.pdf) · [中文 README](README.md)

---

## ✨ Final Results (DashScope, 4-run average)

| Metric             | Value      | Notes                                          |
| ------------------ | ---------- | ---------------------------------------------- |
| **DEV accuracy**   | **85.2%**  | Official test set, 4 runs: 85.3/85.3/85.0/85.0 |
| OOD (mock)         | 82.0%      | Self-built 52-class cross-domain set           |
| MCQ (mock)         | 87.2%      | Self-built balanced A/B/C/D set                |
| **Weighted est.**  | **83.7%**  | Official weights DEV 20% / OOD 60% / MCQ 20%   |
| Avg prompt tokens  | 1,426      | Well below the 2,048 budget                    |
| Anti-injection     | **9 / 9**  | Blocks all 9 injection vectors in mock         |

---

## 🧠 Approach

A **retrieval-based few-shot harness** that builds external memory and indexes during `update(text, label)`, then assembles task-specific prompts in `predict(text)`:

- **Retrieval**: **RRF fusion** of cosine + BM25 to pick top-k neighbors from memory
- **Prompt construction**: few-shot exemplars + LLM-generated label-semantic descriptions + anti-injection wrapping
- **Task routing**: heuristic MCQ vs. classification detection from training text, with different system messages and reasoning modes per task type
- **Robustness**: fall back to top-1 retrieval on LLM failure; MCQ detection via option-stem scan instead of brittle heuristics

Full evolution (P1 baseline → 8 failed experiments → 4 reversals → 2 hardening steps) with numbers is documented in [探索报告.md](探索报告.md).

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `openai`, `transformers`, `numpy`.

### 2. Configure your LLM API key

Edit [llm_client.py](llm_client.py):

```python
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY  = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"   # your key here
MODEL    = "qwen3-8b"
```

> Defaults to Alibaba DashScope (Qwen3-8B, thinking disabled). A SiliconFlow fallback is commented in the same file.

### 3. Run local evaluation

```bash
python run.py                          # default: 4 runs, workers=20
python run.py --workers 100            # more concurrency
python run.py --runs 1                 # single run for debugging
```

The script loads `data/train_dev.jsonl` as the training stream and `data/test_dev.jsonl` as the dev set, printing per-run accuracy, token usage, and total elapsed time.

---

## 📁 Project Layout

```text
.
├── solution.py              # Main submission: MyHarness implementation
├── harness_base.py          # Official base class (read-only)
├── llm_client.py            # LLM call + tokenizer wrapper
├── run.py                   # Local evaluation script
├── run_test.sh              # One-shot test runner
├── requirements.txt
├── data/                    # DEV / OOD / MCQ datasets (train + test)
├── tokenizer/               # Qwen3 tokenizer (used for local token counting)
├── test/                    # Diagnostic helpers
├── output/                  # Historical run logs + solution snapshots
├── 探索报告.pdf / .md / .tex # Exploration report (20% of the grade)
├── 任务清单.md              # Requirements breakdown
└── 探索清单.md              # Experiment log
```

---

## 🛡️ Hard Constraints

| Constraint              | Value / Rule                                                   |
| ----------------------- | -------------------------------------------------------------- |
| Model                   | Qwen3-8B Instruct (weights frozen, thinking mode off)          |
| Prompt budget           | ≤ 2,048 tokens (tail-truncated if exceeded)                    |
| Allowed libraries       | stdlib + numpy + harness_base (no sklearn / torch / requests)  |
| Persistence             | No disk writes                                                 |
| Task mix                | 77-class intent + OOD + A/B/C/D MCQ + prompt-injection samples |
| Grading                 | 80% objective (weighted avg over 4 runs) + 20% report          |

---

## 📜 License

[MIT](LICENSE) © 2026 kalalsland
