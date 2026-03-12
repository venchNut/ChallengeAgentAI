# Reply Mirror Challenge

Classificazione supporto preventivo per cittadini (0=monitoring, 1=support).

## Setup

```bash
python -m venv venv
.\venv\Scripts\activate
pip install pandas numpy scikit-learn langfuse python-dotenv ulid-py

# Optional per LLM assist:
pip install langchain langchain-openai
```

## Files

- **solve.py** - Sistema completo: risk scoring + ML + Langfuse tracking
- **data_agent.py** - Carica CSV/JSON
- **main.py** - LLM con Langfuse tracking (opzionale)
- **create_submission.py** - Crea ZIP per submission finale

## Usage

```bash
# 1. Training flow (unlimited runs)
python solve.py truman          # train → output_truman_train.txt
python solve.py deus            # train → output_deus_train.txt
python solve.py brave           # train → output_brave_train.txt

# 2. Evaluation flow (ONE SHOT — FINALE, irreversibile!)
python solve.py truman eval     # eval → output_truman.txt
python solve.py deus   eval
python solve.py brave  eval

# 3. Create submission ZIP
python create_submission.py truman
```

## Submission Process

**Per ogni dataset ci sono DUE upload:**

1. **Training submission** (unlimited — per testare)
   - File: `output_<dataset>_train.txt`
   - Session ID: da `session_<dataset>_TRAIN_YYYYMMDD.txt`
   - No ZIP richiesto

2. **Evaluation submission** (ONE submission only — FINALE!)
   - File: `output_<dataset>.txt` + source code ZIP
   - Session ID: da `session_<dataset>_EVAL_YYYYMMDD.txt`
   - ⚠️ Solo la prima submission sarà accettata!

## Sistema

**Approccio:** Risk scoring multi-dimensionale
- 30+ features (temporal, demographic, geospatial, interactions)
- Risk score basato su red flags (low activity, declining trends, erratic behavior)
- Random Forest con class balancing
- Langfuse tracking obbligatorio per ogni run

**Features chiave:**
- Temporal: avg/std activity, sleep, env, trends, declines
- Demographic: age, retirement, senior flags
- Geospatial: mobility ratio, daily moves, low mobility detection
- Derived: erratic behavior, specialist consultations, interactions

## Models (Challenge)

```
qwen/qwen3-30b-a3b-thinking-2507
qwen/qwen3-14b
microsoft/phi-4
```
