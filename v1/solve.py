"""
Solve — Reply Mirror fraud detection pipeline.

Usage:
  python solve.py <level>        → predict on TRAINING dataset (unlimited runs)
  python solve.py <level> eval   → predict on EVALUATION dataset (ONE shot only!)

Architecture:
  DataAgent → build_sender_profiles + build_population_context
  → per-transaction: extract_features + risk_score → ChallengeSystem.assess_transaction
     - Fast path   (risk ≤ 2 or ≥ 14): 1 LLM call
     - Coop path   (risk 3–13):         TransactionAgent + ContextAgent → DecisionAgent

Output validity (from Rules — submission is INVALID if violated):
  • must not be empty
  • must not contain ALL transactions
  • must identify ≥ 15 % of fraudulent transactions (recall floor)
  → safety guards applied automatically if heuristic scores suggest low coverage.
"""

import sys
import os
import math
import ulid
import pandas as pd
from dotenv import load_dotenv
from langfuse import Langfuse, observe, propagate_attributes
from datetime import datetime

from data_agent import DataAgent
from main import ChallengeSystem

load_dotenv()

langfuse_client = Langfuse(
    public_key  = os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key  = os.getenv("LANGFUSE_SECRET_KEY"),
    host        = os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
)


# ---------------------------------------------------------------------------
# Session ID helpers
# ---------------------------------------------------------------------------

def generate_session_id(level: str, operation: str) -> str:
    """Generate a ULID-based session ID and persist it to file."""
    team_name  = os.getenv("TEAM_NAME", "team")
    session_id = f"{team_name}-{ulid.new().str}"
    date_str   = datetime.now().strftime("%Y%m%d")
    log_file   = f"session_level_{level}_{operation}_{date_str}.txt"
    with open(log_file, "w") as f:
        f.write(session_id)
    print(f"\n{'='*60}")
    print(f"[Langfuse Session ID] {session_id}")
    print(f"[Saved to]            {log_file}")
    print(f"{'='*60}\n")
    return session_id


# ---------------------------------------------------------------------------
# Haversine distance (km)
# ---------------------------------------------------------------------------

def _haversine(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(d_lng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(tx_data: dict, profile: dict) -> dict:
    """
    Build a flat feature dict for one transaction.
    All features have safe defaults so missing data never raises.
    """
    tx      = tx_data["tx"]
    sender  = tx_data["sender"]
    gps_df  = tx_data["gps_near"]
    f: dict = {}

    # --- Transaction core ---
    f["tx_type"]       = str(tx.get("TransactionType", ""))
    f["amount"]        = float(tx.get("Amount", 0))
    f["balance_after"] = float(tx.get("Balance", 0))
    f["balance_negative"] = 1 if f["balance_after"] < 0 else 0
    total_before = f["amount"] + f["balance_after"]
    f["balance_ratio"] = f["amount"] / max(total_before, 1e-9) if total_before > 0 else 1.0
    f["payment_method"] = str(tx.get("PaymentMethod", ""))
    f["tx_location"]    = str(tx.get("Location", "") or "")

    # Description field — free text like "Salary payment Jan", "Rent Q1"
    desc = str(tx.get("Description", "") or "").lower()
    _LEGIT_DESC = ("salary","rent","payroll","subscription","insurance",
                   "utility","mortgage","refund","dividend","reimbursement")
    f["desc_legit"] = 1 if any(k in desc for k in _LEGIT_DESC) else 0
    f["desc_snippet"] = desc[:80]
    f["hour"]     = int(tx["_hour"])       if "_hour"    in tx.index and not pd.isna(tx["_hour"])    else 0
    f["is_night"] = int(tx["_is_night"])   if "_is_night"   in tx.index and not pd.isna(tx["_is_night"]) else 0
    f["is_weekend"] = int(tx["_is_weekend"]) if "_is_weekend" in tx.index and not pd.isna(tx["_is_weekend"]) else 0

    # --- Sender behavioural vs this transaction ---
    if profile:
        mean = profile["amount_mean"]
        std  = max(profile["amount_std"], 1e-9)
        f["sender_amount_mean"] = mean
        f["sender_amount_std"]  = profile["amount_std"]
        f["amount_zscore"]      = (f["amount"] - mean) / std
        f["sender_tx_count"]    = profile["tx_count"]
        f["unusual_type"]   = 1 if f["tx_type"] and f["tx_type"] != profile["common_type"] else 0
        f["unusual_method"] = 1 if f["payment_method"] and f["payment_method"] != profile["common_method"] else 0
        rid = str(tx.get("RecipientID") or "")
        f["recipient_new"]  = 1 if rid and rid not in profile["known_recipients"] else 0
        f["phishing_sms"]   = profile.get("phishing_score", 0)
        f["phishing_email"] = 0   # communication split computed below
    else:
        f["sender_amount_mean"] = f["amount"]
        f["sender_amount_std"]  = 0.0
        f["amount_zscore"]      = 0.0
        f["sender_tx_count"]    = 1
        f["unusual_type"]   = 0
        f["unusual_method"] = 0
        f["recipient_new"]  = 0
        f["phishing_sms"]   = 0
        f["phishing_email"] = 0

    # --- Communication snippets ---
    sms   = tx_data.get("sms", "")
    email = tx_data.get("email", "")
    audio = tx_data.get("audio", {})
    f["sms_snippet"]    = sms[:300]   if sms   else ""
    f["email_snippet"]  = email[:300] if email else ""
    f["audio_snippet"]  = audio.get("snippet", "")[:300]
    f["audio_phishing"] = audio.get("phishing_score", 0)
    f["audio_calls"]    = audio.get("call_count", 0)
    # Phishing email independent score
    if email:
        from data_agent import DataAgent as _DA
        f["phishing_email"] = _DA._phishing_score(email)

    # --- GPS consistency ---
    if gps_df is not None and len(gps_df) > 0 and sender is not None:
        # Best GPS point = closest in time to transaction
        ts = tx["Timestamp"]
        if hasattr(ts, "timestamp"):
            gps_df = gps_df.copy()
            gps_df["_dt"] = (gps_df["timestamp"] - ts).abs()
            best = gps_df.sort_values("_dt").iloc[0]
            gps_lat, gps_lng = float(best["lat"]), float(best["lng"])
            f["has_gps_data"] = 1
            # If we have a residence city + tx_location, check rough plausibility
            # Also check if GPS is far from the transaction's declared location
            # For in-person: tx has lat/lng? No — it has a city name. Use residence.
            res_lat = float(sender.get("residence_lat") or 0)
            res_lng = float(sender.get("residence_lng") or 0)
            if res_lat != 0:
                dist_gps_home = _haversine(gps_lat, gps_lng, res_lat, res_lng)
                f["gps_distance_km"] = round(dist_gps_home, 1)
                # Flag if GPS is far from home (> 200 km unusual for day-to-day)
                f["gps_match"] = "YES" if dist_gps_home < 200 else "NO"
            else:
                f["gps_distance_km"] = 0
                f["gps_match"] = "NO_DATA"
        else:
            f["has_gps_data"] = 0
            f["gps_distance_km"] = 0
            f["gps_match"] = "NO_DATA"
    else:
        f["has_gps_data"] = 0
        f["gps_distance_km"] = 0
        f["gps_match"] = "NO_DATA"

    # --- Sender demographics ---
    if sender is not None:
        f["sender_age"]  = int(sender.get("age", 0) or 0)
        f["sender_city"] = str(sender.get("residence_city", "") or "")
    else:
        f["sender_age"]  = 0
        f["sender_city"] = ""

    return f


# ---------------------------------------------------------------------------
# Heuristic risk score (0–20) — fast prior for dispatch + fallback
# ---------------------------------------------------------------------------

def calculate_risk_score(features: dict) -> int:
    """
    Compute a heuristic risk score (0–20).
    High scores → almost certainly fraudulent  → fast path (1).
    Low scores  → almost certainly legitimate  → fast path (0).
    Mid scores  → ambiguous                    → LLM cooperative path.
    """
    risk = 0

    # Amount anomaly relative to sender's history
    z = features.get("amount_zscore", 0)
    if z > 5:
        risk += 5
    elif z > 3:
        risk += 3
    elif z > 2:
        risk += 1

    # Balance impact
    if features.get("balance_negative", 0):
        risk += 3
    elif features.get("balance_ratio", 0) > 0.9:
        risk += 2

    # Timing
    if features.get("is_night", 0):
        risk += 2
    if features.get("is_night", 0) and features.get("is_weekend", 0):
        risk += 1

    # Recipient unknown
    if features.get("recipient_new", 0):
        risk += 2

    # Unusual behavior for this sender
    if features.get("unusual_type", 0):
        risk += 1
    if features.get("unusual_method", 0):
        risk += 1

    # Communication phishing signals
    phishing = features.get("phishing_sms", 0) + features.get("phishing_email", 0)
    if phishing >= 4:
        risk += 3
    elif phishing >= 2:
        risk += 2
    elif phishing >= 1:
        risk += 1

    # Audio call phishing signals (new for levels 4+)
    audio_phish = features.get("audio_phishing", 0)
    if audio_phish >= 3:
        risk += 3
    elif audio_phish >= 2:
        risk += 2
    elif audio_phish >= 1:
        risk += 1

    # GPS inconsistency
    if features.get("gps_match") == "NO":
        risk += 2
        if features.get("gps_distance_km", 0) > 500:
            risk += 1

    # First-time sender (could be legitimate or fraudulent account)
    if features.get("sender_tx_count", 0) <= 1:
        risk += 1

    # Description signal: clearly legit descriptions reduce risk
    if features.get("desc_legit", 0):
        risk = max(0, risk - 2)

    return min(risk, 20)


# ---------------------------------------------------------------------------
# Population context — computed once per run
# ---------------------------------------------------------------------------

def build_population_context(tx_ids, agent: DataAgent, profiles: dict) -> tuple:
    """
    Scan all transactions once to produce:
      pop_stats         — amount mean/std, night_rate (for TransactionAgent context)
      pop_context       — human-readable distribution summary for DecisionAgent
      fallback_threshold — p70 of risk scores, clamped [4, 12]
    """
    all_features = []
    for tid in tx_ids:
        tx_data = agent.get_transaction_data(tid)
        sender_id = str(tx_data["tx"]["SenderID"])
        profile   = profiles.get(sender_id)
        f = extract_features(tx_data, profile)
        f["_risk"] = calculate_risk_score(f)
        all_features.append(f)

    n = len(all_features)
    pop_stats = {
        "amount_mean": sum(f["amount"]      for f in all_features) / n,
        "amount_std":  (sum((f["amount"] - sum(ff["amount"] for ff in all_features)/n)**2
                            for f in all_features) / n) ** 0.5,
        "night_rate":  sum(f["is_night"]    for f in all_features) / n,
    }

    risks = sorted(f["_risk"] for f in all_features)
    p25 = risks[max(0, int(n * 0.25) - 1)]
    p50 = risks[max(0, int(n * 0.50) - 1)]
    p75 = risks[max(0, int(n * 0.75) - 1)]
    p70_idx = max(0, int(n * 0.70) - 1)
    fallback_threshold = max(4, min(12, risks[p70_idx]))

    pop_context = (
        f"Population context ({n} transactions): "
        f"risk scores p25={p25}, p50={p50}, p75={p75} (scale 0–20). "
        f"Amount: mean={pop_stats['amount_mean']:.2f}, std={pop_stats['amount_std']:.2f}. "
        f"Night transactions: {pop_stats['night_rate']:.1%}. "
        f"A transaction at p75 risk or above is among the highest-risk in this batch."
    )

    print(f"Population: n={n}, amount_mean={pop_stats['amount_mean']:.2f}, "
          f"risk p25/p50/p75={p25}/{p50}/{p75}, fallback_threshold={fallback_threshold}")

    return pop_stats, pop_context, fallback_threshold, all_features


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------

@observe()
def predict(level: str, is_eval: bool = False) -> list:
    """
    Full pipeline for one dataset level.
    Returns list of fraudulent Transaction IDs.
    """
    operation = "EVAL" if is_eval else "TRAIN_PREDICT"

    # Guard — eval is irreversible
    if is_eval:
        date_str = datetime.now().strftime("%Y%m%d")
        eval_session_file = f"session_level_{level}_EVAL_{date_str}.txt"
        if os.path.exists(eval_session_file):
            print(f"\n{'!'*60}")
            print(f"WARNING: {eval_session_file} already exists.")
            print("EVAL submission is FINAL and IRREVERSIBLE.")
            print(f"{'!'*60}")
            confirm = input("Regenerate anyway? [y/N] ").strip().lower()
            if confirm != "y":
                print("Aborted. Use the existing session file for submission.")
                sys.exit(0)

    session_id = generate_session_id(level.upper(), operation)

    with propagate_attributes(
        session_id = session_id,
        trace_name = f"{level.upper()}_{operation}",
        metadata   = {"dataset": level, "operation": operation, "is_eval": str(is_eval)},
    ):
        dataset_path = _resolve_dataset_path(level, is_eval)
        print(f"Dataset: {dataset_path}")

        agent = DataAgent(dataset_path)
        agent.load_all_data()

        tx_ids   = agent.transactions_df["TransactionID"].tolist()
        n_total  = len(tx_ids)
        print(f"Transactions to assess: {n_total}")

        # Pre-compute sender profiles (one pass over all transactions)
        profiles = agent.build_sender_profiles()
        print(f"Sender profiles built: {len(profiles)}")

        # Pre-compute population context (second pass — builds features + risk scores)
        print("Building population context …")
        pop_stats, pop_context, fallback_threshold, all_features = \
            build_population_context(tx_ids, agent, profiles)

        # Build a fast-lookup: tx_id → pre-computed features + risk
        feature_map = {tx_ids[i]: all_features[i] for i in range(n_total)}

        # All (tx_id, risk_score) pairs for validity guards at the end
        all_risks = [(tid, feature_map[tid]["_risk"]) for tid in tx_ids]

        system = ChallengeSystem()
        fraud_ids  = []
        errors     = 0

        for idx, tid in enumerate(tx_ids, 1):
            features   = feature_map[tid]
            risk_score = features["_risk"]

            try:
                decision = system.assess_transaction(
                    session_id, features, risk_score,
                    pop_context=pop_context,
                )
                if decision == 1:
                    fraud_ids.append(tid)
                label = "→ FRAUD" if decision == 1 else "→ legit"
                print(f"  [{idx:>5}/{n_total}] {tid[:36]}  risk={risk_score:>2}  {label}")
            except Exception as exc:
                errors += 1
                fallback = 1 if risk_score >= fallback_threshold else 0
                if fallback == 1:
                    fraud_ids.append(tid)
                label = "→ FRAUD" if fallback else "→ legit"
                print(
                    f"  [{idx:>5}/{n_total}] {tid[:36]}  risk={risk_score:>2}  "
                    f"[HEURISTIC] {label}  ({str(exc)[:80]})"
                )

        system.langfuse.flush()
        langfuse_client.flush()

        # ------------------------------------------------------------------
        # Output validity guards (Rules §3)
        # ------------------------------------------------------------------

        # Guard 1: must not be empty
        # Minimum floor: at least 5% of transactions for recall safety
        min_floor = max(1, int(0.05 * n_total))
        if len(fraud_ids) < min_floor:
            print(f"\n[GUARD] Only {len(fraud_ids)} fraud IDs — boosting to {min_floor} "
                  f"using top risk scores")
            already = set(fraud_ids)
            for tid_r, _ in sorted(all_risks, key=lambda x: -x[1]):
                if len(fraud_ids) >= min_floor:
                    break
                if tid_r not in already:
                    fraud_ids.append(tid_r)

        # Guard 2: must not flag everything
        if len(fraud_ids) >= n_total:
            print(f"\n[GUARD] All {n_total} transactions flagged — capping at 50%")
            fraud_ids = fraud_ids[: n_total // 2]

        # ------------------------------------------------------------------
        # Write output
        # ------------------------------------------------------------------
        output_file = (
            f"output_{level}.txt"
            if is_eval
            else f"output_{level}_train.txt"
        )
        with open(output_file, "w", encoding="ascii") as f:
            for tid in fraud_ids:
                f.write(f"{tid}\n")

        pct = len(fraud_ids) / n_total * 100 if n_total else 0
        print(f"\nOutput  : {output_file}")
        print(f"Fraud   : {len(fraud_ids)}/{n_total} ({pct:.1f}% flagged)")
        if errors:
            print(f"Heuristic fallback used for {errors} transaction(s)")

        return fraud_ids


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dataset path resolver
# ---------------------------------------------------------------------------

_DATASET_MAP = {
    "truman": "The+Truman+Show",
    "deus":   "Deus+Ex",
    "brave":  "Brave+New+World",
    "1984":   "1984",
    "blade":  "Blade+Runner",
}

def _resolve_dataset_path(name: str, is_eval: bool) -> str:
    """
    Map a short dataset name to its actual folder path.

    Old structure (levels 1-3) — nested double folder:
      truman → The+Truman+Show_train/The Truman Show_train/public

    New structure (levels 4-5) — flat single folder:
      1984   → 1984_eval/public
      blade  → Blade Runner_eval/public
    """
    key = name.lower()
    if key not in _DATASET_MAP:
        raise ValueError(f"Unknown dataset '{name}'. Choose: {', '.join(_DATASET_MAP)}")
    folder = _DATASET_MAP[key]
    suffix = "eval" if is_eval else "train"

    # New-style flat layout: 1984, blade
    if key in ("1984", "blade"):
        folder_plain = folder.replace("+", " ")
        flat = f"{folder_plain}_{suffix}/public"
        if os.path.isdir(flat):
            return flat
        # Fallback: maybe extracted with + in name
        return f"{folder}_{suffix}/public"

    # Old-style nested layout: truman, deus, brave
    folder_plain = folder.replace("+", " ")
    return f"{folder}_{suffix}/{folder_plain}_{suffix}/public"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python solve.py <dataset> [eval]")
        print("  python solve.py truman        → TRAIN (unlimited)")
        print("  python solve.py truman eval   → EVAL  (ONE SHOT — irreversible!)")
        print("  Datasets: truman | deus | brave")
        sys.exit(1)

    level   = sys.argv[1]
    is_eval = "eval" in sys.argv

    predict(level, is_eval=is_eval)
