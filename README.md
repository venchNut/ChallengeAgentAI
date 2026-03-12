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
# 1. Training flow (test sul training dataset - ripetibile)
python solve.py 1              # train + predict → output_level_1_train.txt

# 2. Evaluation flow (submission FINALE - solo UNA volta!)
python solve.py 1 eval         # predict → output_level_1.txt

# 3. Create submission ZIP (per eval dataset)
python create_submission.py 1  # crea ZIP + mostra Session ID

# Con LLM assist per casi borderline
python solve.py 1 eval --llm
```

## Submission Process

**Per ogni livello ci sono DUE upload:**

1. **Training submission** (unlimited submissions per testing)
   - File: `output_level_X_train.txt`
   - Session ID: da `session_level_X_TRAIN_PREDICT_YYYYMMDD.txt`
   - No ZIP richiesto
   
2. **Evaluation submission** (ONE submission only - FINALE!)
   - File: `submission_level_X_YYYYMMDD.zip` (contiene output + source code)
   - Session ID: da `session_level_X_EVAL_YYYYMMDD.txt`
   - ⚠️ ATTENZIONE: solo la prima submission sarà accettata!

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
