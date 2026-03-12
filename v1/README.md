# MirrorPay Fraud Detector — v1 (OOP)

Autonomous fraud detection system for the **Reply Mirror** challenge (year 2087).
Classifies each financial transaction as `0` (legit) or `1` (fraud).

Datasets: **The Truman Show** · **Deus Ex** · **Brave New World**

---

## Project Structure

| File | Lines | Role |
|---|---|---|
| `solve.py` | 478 | Main pipeline: session mgmt, feature engineering, risk scoring, orchestration, output guards |
| `main.py` | 212 | `ChallengeSystem` class: LLM agent dispatch, model cascade, Langfuse tracking |
| `data_agent.py` | 366 | Data loader: CSV/JSON parsing, GPS correlation, SMS/mail linking, sender profiles |
| `create_submission.py` | 100 | Packages output + source into submission ZIP |
| `create_all_submissions.py` | 35 | Batch submission for all datasets |
| `check_trace.py` | 142 | Query Langfuse to verify session traces exist |

---

## Setup

```bash
cd v1
python -m venv ../venv
source ../venv/bin/activate
pip install pandas numpy langfuse python-dotenv ulid-py langchain langchain-openai
```

**`.env`** (nella root del workspace, condivisa con v2):
```
OPENROUTER_API_KEY=sk-or-v1-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://challenges.reply.com/langfuse
TEAM_NAME=PowerBiBabi
```

---

## Usage

```bash
# Training (illimitato — per iterare e testare)
python solve.py truman          # → output_truman_train.txt
python solve.py deus            # → output_deus_train.txt
python solve.py brave           # → output_brave_train.txt

# Evaluation (UNA SOLA VOLTA — irreversibile!)
python solve.py truman eval     # → output_truman.txt
python solve.py deus   eval
python solve.py brave  eval

# Verifica sessione su Langfuse
python check_trace.py <SESSION_ID>

# Crea ZIP per submission
python create_submission.py truman
```

Ogni run salva `session_level_<DATASET>_<TRAIN_PREDICT|EVAL>_YYYYMMDD.txt` con il Session ID Langfuse necessario per il submit.

---

## Architecture

### Pipeline Overview

```
solve.py
  │
  ├─ DataAgent.load_all_data()
  │    transactions.csv  →  parsed, typed, time-enriched (_hour, _is_night, _is_weekend)
  │    users.json        →  age (2087 - birth_year), residence lat/lng/city
  │    locations.json    →  GPS points (BioTag → user_id, normalized)
  │    sms.json          →  "Hi FirstName" → matched to user IBAN
  │    mails.json        →  "To: FirstName LastName" → matched to user IBAN
  │
  ├─ build_sender_profiles()          ← one pass: per-sender behavioral stats
  │    tx_count, amount_mean/std/max, common_type, common_method,
  │    night_rate, known_recipients, phishing_score (SMS + mail keywords)
  │
  ├─ build_population_context()       ← one pass: population-level stats
  │    amount mean/std, night_rate, risk percentiles (p25/p50/p75),
  │    fallback_threshold (p70, clamped [4,12])
  │
  └─ per-transaction loop:
       extract_features()  →  30+ features flat dict
       calculate_risk_score()  →  heuristic 0–20
       ChallengeSystem.assess_transaction()
            │
            ├── risk ≤ 4   → return 0 (legit) — ZERO LLM calls
            ├── risk ≥ 15  → return 1 (fraud) — ZERO LLM calls
            └── risk 5–14  → DecisionAgent    — exactly 1 LLM call
```

### Feature Set (30+ features)

**Transaction core:** tx_type, amount, balance_after, balance_negative, balance_ratio, payment_method, tx_location, desc_legit, desc_snippet, hour, is_night, is_weekend

**Sender behavioral:** sender_amount_mean/std, amount_zscore, sender_tx_count, unusual_type, unusual_method, recipient_new

**Communication:** phishing_sms (keyword score 0–3), phishing_email (keyword score 0–3), sms_snippet, email_snippet

**GPS:** has_gps_data, gps_distance_km (haversine from home), gps_match (YES if < 200km)

**Demographics:** sender_age, sender_city

### Risk Score Components (0–20)

| Signal | Points |
|---|---|
| amount_zscore > 5 / > 3 / > 2 | +5 / +3 / +1 |
| Negative balance | +3 |
| balance_ratio > 0.9 | +2 |
| Night (00–05) | +2 |
| Night + weekend | +1 |
| New recipient | +2 |
| Unusual tx type | +1 |
| Unusual payment method | +1 |
| Phishing signals (sms+mail) ≥ 4 / ≥ 2 / ≥ 1 | +3 / +2 / +1 |
| GPS mismatch (> 200km) | +2 |
| GPS distance > 500km | +1 |
| First-time sender (≤ 1 tx) | +1 |
| Legit description keyword | −2 |

### LLM Decision Agent

The `DecisionAgent` receives a compact single-line prompt:
```
risk=X/20 | Population context (...) | type=... amt=... z=... bal=... night=... gps=... phish=... | 1 or 0:
```

**System prompt:** "MirrorPay fraud adjudicator. Output ONLY '1' (fraud) or '0' (legit). FN >> FP. When uncertain → 1."

**Response parsing:** First `0` or `1` char found. Semantic fallback: "fraud/suspicious/flag" → 1. Default → 0.

### Model Cascade

| Priority | Model | Note |
|---|---|---|
| 1 | `google/gemma-3-4b-it` | Primary |
| 2 | `nvidia/nemotron-nano-9b-v2` | Fallback |
| 3 | `microsoft/phi-4` | Last resort |

Each model gets 2 attempts (retry on 429 with 15s wait). If all 3 fail → heuristic fallback using `fallback_threshold` (p70 of risk scores).

### Langfuse Tracking

- **Session:** 1 ULID per run via `propagate_attributes(session_id=...)` wrapping the entire tx loop
- **Traces:** `@observe()` on `predict()` (solve.py) + `_call_model()` (main.py)
- **LLM:** `CallbackHandler()` inside `_call_model()` captures tokens, cost, latency
- **Flush:** `system.langfuse.flush()` + `langfuse_client.flush()` at end

### Output Validity Guards (from Rules §3)

1. **Non-empty:** If < 5% flagged → boost with highest risk scores
2. **Not all:** If 100% flagged → cap at 50%
3. **Recall floor:** Rules require ≥ 15% of real fraud detected (enforced by conservative scoring bias)

---

## Scoring (from Rules §4)

- **Primary:** Accuracy — balanced fraud detection vs false positive avoidance
- **Secondary:** Cost, speed, efficiency — operational sustainability
- **Asymmetric:** FN (missed fraud) costs MORE than FP (blocked legit)

---

## Possible Optimizations

### Accuracy Improvements
1. **Widen the ambiguous zone** — ZERO_LO=4 → 3, ZERO_HI=15 → 16 to let the LLM handle more borderline cases (costs more tokens but catches edge cases)
2. **Population-relative thresholds** — Compute ZERO_LO/ZERO_HI dynamically from the risk distribution (e.g., p10 and p90) instead of hardcoded
3. **Multi-call cooperative path for mid-risk** — Re-enable TransactionAgent + ContextAgent for risk 7–12 (the most uncertain zone). Currently eliminated for cost but may improve accuracy in harder datasets
4. **Few-shot examples in prompt** — Include 2 fraud + 2 legit anchoring examples from population extremes (code for this existed in old version)
5. **Temporal features** — Add time-since-last-tx for sender, burst detection (N tx in last hour), velocity checks
6. **Cross-sender signals** — Flag recipients that receive from many first-time senders (mule account pattern)

### Cost Reductions
7. **Batch similar transactions** — Group identical risk profiles and send one LLM call per group instead of per transaction
8. **Tighter max_tokens** — Currently 50; if models reliably output just "1" or "0", reduce to 5-10
9. **Cache identical prompts** — Hash (system+user) and skip LLM if seen before (common in low-variance datasets)

### Robustness
10. **Retry with exponential backoff** — Currently fixed 15s on 429; exponential (5s → 15s → 45s) handles sustained rate limits better
11. **Langfuse async flush** — Current `.flush()` blocks; schedule periodic flushes during the loop so traces arrive even if the process crashes
12. **Checkpointing** — Save partial results every 100 tx so a crash doesn't lose everything

### Data Quality
13. **SMS/mail linking recall** — Current "Hi FirstName" regex misses messages that don't start with a greeting (e.g. "Dear Customer"). Add fallback patterns
14. **GPS window tuning** — ±2h window is arbitrary; for fast-moving fraud (e.g., 2 in-person payments in different cities within 30 min), a tighter window with distance checks would be more discriminative
