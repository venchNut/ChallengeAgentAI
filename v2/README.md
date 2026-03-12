# MirrorPay Fraud Detector — v2 (Functional)

Autonomous fraud detection system for the **Reply Mirror** challenge (year 2087).
Functional architecture — no classes, flat modules.

Datasets: **The Truman Show** · **Deus Ex** · **Brave New World**

---

## Project Structure

| File | Lines | Role |
|---|---|---|
| `run.py` | 265 | Main pipeline: session mgmt, feature extraction, risk scoring, orchestration, output guards |
| `agents.py` | 108 | LLM agent functions: model cascade, `@observe` tracking, verdict logic |
| `loader.py` | 193 | Data loading: CSV/JSON, GPS, SMS/mail linking, sender profiles |

---

## Setup

```bash
cd v2
source ../venv/bin/activate   # shared venv with v1
# Dependencies already installed — same requirements
```

**`.env`** (nella root del workspace):
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
# Training (illimitato)
python run.py truman          # → output_truman_train.txt
python run.py deus            # → output_deus_train.txt
python run.py brave           # → output_brave_train.txt

# Evaluation (UNA SOLA VOLTA — irreversibile!)
python run.py truman eval     # → output_truman.txt
python run.py deus   eval
python run.py brave  eval
```

Ogni run salva `session_<dataset>_<TRAIN|EVAL>_YYYYMMDD.txt` con il Session ID.

---

## Architecture

### Pipeline Overview

```
run.py
  │
  ├─ loader.load(dataset, eval_mode)
  │    transactions.csv  →  parsed, typed, time columns (_h, _ni, _we)
  │    users.json        →  age, res_lat, res_lng, res_city
  │    locations.json    →  GPS (user_id, ts, lat, lng)
  │    sms.json          →  linked by "Hi FirstName" → IBAN
  │    mails.json        →  linked by "To: First Last" → IBAN
  │
  ├─ loader.build_profiles(tx, sms, mails)   ← per-sender stats
  │    n, amt_mean/std/max, top_type, top_method, night_rate,
  │    recipients (set), phish (keyword score 0–3)
  │
  ├─ pre-compute all features + risks → population context string
  │    risk percentiles (p25/p50/p75), avg amount  
  │
  └─ per-transaction loop:
       features(ctx, profiles)  →  flat feature dict
       risk(features)  →  heuristic 0–20
       agents.assess(session_id, features, risk)
            │
            ├── risk ≤ 4   → return 0 (legit) — ZERO LLM calls
            ├── risk ≥ 15  → return 1 (fraud) — ZERO LLM calls
            └── risk 5–14  → fast_verdict()   — exactly 1 LLM call
```

### Key Differences from v1

| Aspect | v1 (OOP) | v2 (Functional) |
|---|---|---|
| Style | `ChallengeSystem` class | Pure functions |
| Agent dispatch | `assess_transaction()` method | `agents.assess()` function |
| Data loader | `DataAgent` class with methods | `loader.load()` + `loader.build_profiles()` |
| Session file naming | `session_level_<DATASET>_<OP>_date.txt` | `session_<dataset>_<OP>_date.txt` |
| Docstring comments | docstring-heavy, OOP conventions | minimal, compact |
| Module-level Langfuse | in class `__init__` | module-level `lf = Langfuse(...)` |

Both projects share: same feature set, same risk scoring, same model cascade, same zero-LLM zones, same output guards.

### Feature Set

**Transaction:** tx_type, amount, balance, neg_bal, hour, night, wknd, method, desc_legit, desc_snippet

**Sender:** amt_mean/std, z-score, n_tx, unusual_method, new_rec

**Communication:** phish_sms, phish_mail (keyword score 0–3), sms_snip, mail_snip

**GPS:** gps_km (haversine from residence), gps_match (YES/NO/NO_DATA)

**Demographics:** age, city

### Risk Score (0–20)

| Signal | Points |
|---|---|
| z-score > 5 / > 3 / > 2 | +5 / +3 / +1 |
| Negative balance | +3 |
| Night (00–05) | +2 |
| Night + weekend | +1 |
| New recipient | +2 |
| Unusual payment method | +1 |
| First-time sender | +1 |
| Phishing ≥ 4 / ≥ 2 / ≥ 1 | +3 / +2 / +1 |
| GPS mismatch (> 200km) | +2 |
| GPS > 500km | +1 |
| Legit description | −2 |

### LLM FastVerdict Agent

Compact prompt:
```
risk=X/20 | type=... amt=... z=... bal=... night=... gps=... phish=... | 1 or 0:
```

System: "MirrorPay adjudicator. Output ONLY '1' (fraud) or '0' (legit). FN>>FP. When uncertain → 1."

Parsing: first `0`/`1` char. Fallback: risk > 9 → 1, else → 0.

### Model Cascade

| Priority | Model |
|---|---|
| 1 | `google/gemma-3-4b-it` |
| 2 | `nvidia/nemotron-nano-9b-v2` |
| 3 | `microsoft/phi-4` |

2 attempts per model, 15s wait on 429. All fail → heuristic (flag if above median risk and ≥ 6).

### Langfuse Tracking

- **Session:** 1 ULID via `propagate_attributes(session_id=sid)` wrapping entire pipeline
- **Traces:** `@observe()` on `run()` (run.py) + `_call()` (agents.py)
- **LLM:** `CallbackHandler()` inside `_call()` captures tokens, cost, latency
- **Flush:** `agents.lf.flush()` + `_lf.flush()` at end

### Output Validity Guards

1. **5% floor:** If fewer than 5% flagged → boost with highest risk scores
2. **Cap 50%:** If 100% flagged → cap at 50%

---

## Scoring (from Rules §4)

- **Primary:** Balanced fraud detection (recall + precision)
- **Secondary:** Cost, speed, efficiency
- **Asymmetric costs:** FN >> FP

---

## Possible Optimizations

### Accuracy
1. **Dynamic zero-LLM thresholds** — Compute from risk distribution per dataset instead of fixed 4/15. Harder datasets may need tighter zones
2. **Re-enable cooperative 3-agent path** — For risk 7–12 (highest uncertainty): Reasoner → Sceptic → Verdict. More tokens but better on ambiguous cases
3. **Population-calibrated prompt** — Include pop_context (risk percentiles + avg amount) in the FastVerdict prompt. Currently computed but not passed through
4. **Temporal burst detection** — Count sender's tx in last 1h/6h/24h. Rapid consecutive transactions to new recipients is a strong fraud signal
5. **Recipient graph analysis** — Flag recipients receiving from > 5 distinct first-time senders (mule accounts)

### Cost
6. **Prompt deduplication** — Hash (system+user) prompt per risk bucket; identical features = cached answer
7. **Lower max_tokens** — From 50 → 5 if response is consistently just "0"/"1"
8. **Async batch calls** — Send N LLM requests concurrently (requires httpx/aiohttp) for ~10x throughput

### Robustness
9. **Exponential backoff** — Replace fixed 15s retry with 5s → 15s → 45s
10. **Checkpoint every 100 tx** — Save partial fraud_ids list so crashes don't lose progress
11. **Langfuse periodic flush** — Flush every 50 tx instead of only at end — protects against crash/timeout data loss

### Data Quality
12. **Better SMS/mail linking** — "Hi FirstName" regex misses threads without greetings. Add "Dear Customer", email address matching, phone number extraction
13. **GPS velocity check** — Two GPS points imply max travel speed. If sender GPS at city A at t-30min and tx is in city B 500km away → strong fraud signal
14. **Description NLP** — Beyond keyword match: use the LLM to classify description plausibility (e.g., "emergency transfer" at 3am to new recipient)

1. Run `python run.py <dataset> eval`
2. Upload `output_<dataset>.txt` + source code ZIP
3. Use the session ID from `session_<dataset>_EVAL_YYYYMMDD.txt`
