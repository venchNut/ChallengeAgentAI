# MirrorPay Fraud Detector — v2

Cooperative multi-agent fraud detection for the Reply Mirror challenge.  
Datasets: **The Truman Show** · **Deus Ex** · **Brave New World**

## Setup

```bash
cd v2
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # fill in your keys
```

`.env` keys needed:
```
OPENROUTER_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://challenges.reply.com/langfuse
TEAM_NAME=YourTeamName
```

## Usage

```bash
# Training runs (unlimited — iterate freely)
python run.py truman            # → output_truman_train.txt
python run.py deus              # → output_deus_train.txt
python run.py brave             # → output_brave_train.txt

# Evaluation run (ONE SHOT — IRREVERSIBLE, submit this)
python run.py truman eval       # → output_truman.txt
python run.py deus   eval       # → output_deus.txt
python run.py brave  eval       # → output_brave.txt
```

Each run saves a `session_<dataset>_<TRAIN|EVAL>_YYYYMMDD.txt` with the Langfuse session ID needed for submission.

## Architecture

```
all transactions
      │
      ▼
 loader.py   ←  transactions.csv + users.json
              +  locations.json + sms.json + mails.json
      │
      ▼
build_profiles()  →  per-sender behavioral statistics
      │
      ▼
features() + risk()  →  heuristic score 0–20
      │
      ├── score 0–1   → heuristic 0  (no LLM call — saves tokens)
      ├── score 18–20 → heuristic 1  (no LLM call — saves tokens)
      ├── score 2–3 / 15–17 → FastVerdict  (1 LLM call)
      └── score 4–14  → ReasonerAgent → ScepticAgent → VerdictAgent (3 calls)
```

**Agent roles:**
- **Reasoner** — holistic analysis of all features (suspicious + reassuring)
- **Sceptic** — devil's advocate: challenges the Reasoner's conclusion
- **Verdict** — reads both sides, outputs `0` or `1` (recall-biased)

## Models (priority order)

```
qwen/qwen3-30b-a3b-thinking-2507    ← primary (strongest reasoning)
qwen/qwen3-14b                      ← fallback
microsoft/phi-4                     ← last resort
```

Rate-limited models are retried once after 15 s before cascading to the next.

## Submission

⚠️ **EVAL is final — only the first submission is accepted.**

1. Run `python run.py <dataset> eval`
2. Upload `output_<dataset>.txt` + source code ZIP
3. Use the session ID from `session_<dataset>_EVAL_YYYYMMDD.txt`
