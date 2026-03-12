"""
run.py — MirrorPay fraud detector v2
usage:
  python run.py truman          # train set, unlimited runs
  python run.py truman eval     # eval set, ONE SHOT FINAL
  python run.py deus
  python run.py brave
"""

import sys, os, math, ulid, pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langfuse import Langfuse, observe, propagate_attributes

load_dotenv()  # must run before any module-level Langfuse/OpenRouter init

import loader
import agents

_lf = Langfuse(
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key = os.getenv("LANGFUSE_SECRET_KEY"),
    host       = os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
)


# ── Session ID ────────────────────────────────────────────────────────────────

def new_session(dataset: str, op: str) -> str:
    team = os.getenv("TEAM_NAME", "team")
    sid  = f"{team}-{ulid.new().str}"
    fname = f"session_{dataset}_{op}_{datetime.now().strftime('%Y%m%d')}.txt"
    open(fname, "w").write(sid)
    print(f"\n{'─'*55}\nSession : {sid}\nSaved   : {fname}\n{'─'*55}\n")
    return sid


# ── Haversine ─────────────────────────────────────────────────────────────────

def _hav(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    dl = math.radians(lat2-lat1); dn = math.radians(lng2-lng1)
    a  = math.sin(dl/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dn/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ── Feature extraction ────────────────────────────────────────────────────────

def features(ctx: dict, prof: dict) -> dict:
    row    = ctx["row"]
    sender = ctx["sender"]
    gps    = ctx["gps"]
    sms    = ctx["sms"]
    mail   = ctx["mail"]
    sid    = str(row["SenderID"])
    p      = prof.get(sid, {})

    amt = float(row["Amount"])
    bal = float(row["Balance"])
    std = max(p.get("amt_std", 0), 1e-9)

    f: dict = {}
    f["tx_type"] = str(row.get("TransactionType",""))
    f["amount"]  = amt
    f["balance"] = bal
    f["neg_bal"] = 1 if bal < 0 else 0
    f["hour"]    = int(row["_h"])  if not pd.isna(row["_h"])  else 0
    f["night"]   = int(row["_ni"]) if not pd.isna(row["_ni"]) else 0
    f["wknd"]    = int(row["_we"]) if not pd.isna(row["_we"]) else 0
    f["method"]  = str(row.get("PaymentMethod",""))

    if p:
        f["amt_mean"]      = p["amt_mean"]
        f["amt_std"]       = p["amt_std"]
        f["z"]             = (amt - p["amt_mean"]) / std
        f["n_tx"]          = p["n"]
        f["unusual_method"]= 1 if f["method"] and f["method"] != p.get("top_method","") else 0
        f["new_rec"]       = 1 if str(row.get("RecipientID","")) not in p.get("recipients",set()) else 0
        f["phish_sms"]     = p.get("phish", 0)
        f["phish_mail"]    = loader.phish_score(mail)
    else:
        f["amt_mean"] = amt; f["amt_std"] = 0; f["z"] = 0
        f["n_tx"] = 1; f["unusual_method"] = 0; f["new_rec"] = 0
        f["phish_sms"] = 0; f["phish_mail"] = 0

    f["sms_snip"]  = sms[:200]
    f["mail_snip"] = mail[:200]

    # demographics
    if sender is not None:
        f["age"]  = int(sender.get("age",0) or 0)
        f["city"] = str(sender.get("res_city","") or "")
    else:
        f["age"] = 0; f["city"] = ""

    # GPS
    if len(gps) > 0 and sender is not None:
        ts   = row["Timestamp"]
        gps  = gps.copy()
        gps["_dt"] = (gps["ts"] - ts).abs()
        best = gps.sort_values("_dt").iloc[0]
        rlat = float(sender.get("res_lat") or 0)
        rlng = float(sender.get("res_lng") or 0)
        if rlat != 0:
            d = _hav(float(best["lat"]), float(best["lng"]), rlat, rlng)
            f["gps_km"]    = round(d, 1)
            f["gps_match"] = "YES" if d < 200 else "NO"
        else:
            f["gps_km"] = 0; f["gps_match"] = "NO_DATA"
    else:
        f["gps_km"] = 0; f["gps_match"] = "NO_DATA"

    return f


# ── Risk score ────────────────────────────────────────────────────────────────

def risk(f: dict) -> int:
    r = 0
    z = f.get("z", 0)
    if z > 5:   r += 5
    elif z > 3: r += 3
    elif z > 2: r += 1

    if f.get("neg_bal"):       r += 3
    if f.get("night"):         r += 2
    if f.get("night") and f.get("wknd"): r += 1
    if f.get("new_rec"):       r += 2
    if f.get("unusual_method"):r += 1
    if f.get("n_tx",1) <= 1:  r += 1

    ph = f.get("phish_sms",0) + f.get("phish_mail",0)
    if ph >= 4:   r += 3
    elif ph >= 2: r += 2
    elif ph >= 1: r += 1

    if f.get("gps_match") == "NO":
        r += 2
        if f.get("gps_km",0) > 500: r += 1

    return min(r, 20)


# ── Pipeline ──────────────────────────────────────────────────────────────────

@observe()
def run(dataset: str, eval_mode: bool = False):
    op = "EVAL" if eval_mode else "TRAIN"

    if eval_mode:
        guard = f"session_{dataset}_EVAL_{datetime.now().strftime('%Y%m%d')}.txt"
        if os.path.exists(guard):
            print(f"! {guard} exists — EVAL is FINAL and IRREVERSIBLE !")
            if input("Proceed anyway? [y/N] ").strip().lower() != "y":
                sys.exit(0)

    sid = new_session(dataset, op)

    with propagate_attributes(
        session_id = sid,
        trace_name = f"{dataset.upper()}_{op}",
        metadata   = {"dataset": dataset, "mode": op},
    ):
        data     = loader.load(dataset, eval_mode)
        tx       = data["tx"]
        tx_ids   = tx["TransactionID"].tolist()
        n        = len(tx_ids)
        profiles = loader.build_profiles(tx, data["sms"], data["mails"])
        print(f"profiles: {len(profiles)}  transactions: {n}")

        # population context for reasoner
        risk_scores = []
        pop_ctx     = ""
        fraud, errors = [], 0
        all_risks     = []

        # pre-compute all features + risks for population context
        feat_cache = {}
        for tid in tx_ids:
            ctx = loader.get_tx_context(tid, data)
            f   = features(ctx, profiles)
            r   = risk(f)
            feat_cache[tid] = (f, r)
            risk_scores.append(r)

        if risk_scores:
            rs = sorted(risk_scores)
            nn = len(rs)
            p25, p50, p75 = rs[nn//4], rs[nn//2], rs[3*nn//4]
            amt_vals = [feat_cache[t][0]["amount"] for t in tx_ids]
            avg_amt  = sum(amt_vals)/nn
            pop_ctx  = (f"Population ({nn} tx): risk p25={p25} p50={p50} p75={p75} (scale 0-20). "
                        f"Avg amount={avg_amt:.2f}. "
                        f"Risk ≥p75 means top-quartile suspicion.")
            print(f"pop: p25={p25} p50={p50} p75={p75}")

        for i, tid in enumerate(tx_ids, 1):
            f, r = feat_cache[tid]
            all_risks.append((tid, r))

            try:
                dec = agents.assess(sid, f, r, pop_ctx)
                if dec == 1:
                    fraud.append(tid)
                tag = "FRAUD" if dec else "legit"
                print(f"  [{i:>5}/{n}] {tid[:36]}  r={r:>2}  {tag}")
            except Exception as e:
                errors += 1
                # fallback: flag if above median risk
                median_r = sorted(rv for _,rv in all_risks)[len(all_risks)//2]
                if r >= max(median_r, 6):
                    fraud.append(tid)
                    print(f"  [{i:>5}/{n}] {tid[:36]}  r={r:>2}  [HEURISTIC FRAUD]  {str(e)[:60]}")
                else:
                    print(f"  [{i:>5}/{n}] {tid[:36]}  r={r:>2}  [HEURISTIC legit]  {str(e)[:60]}")

        agents.lf.flush()
        _lf.flush()

        # ── validity guards ────────────────────────────────────────────────
        floor = max(1, int(0.05 * n))  # 5% floor — safer recall guarantee
        if len(fraud) < floor:
            seen = set(fraud)
            for tid, _ in sorted(all_risks, key=lambda x: -x[1]):
                if len(fraud) >= floor: break
                if tid not in seen: fraud.append(tid)
            print(f"[guard] boosted to {len(fraud)} fraud IDs")

        if len(fraud) >= n:
            fraud = fraud[:n//2]
            print("[guard] capped at 50%")

        # ── write output ───────────────────────────────────────────────────
        out = f"output_{dataset}.txt" if eval_mode else f"output_{dataset}_train.txt"
        with open(out, "w", encoding="ascii") as fh:
            for tid in fraud: fh.write(f"{tid}\n")

        pct = len(fraud)/n*100 if n else 0
        print(f"\n{'─'*55}")
        print(f"output : {out}")
        print(f"fraud  : {len(fraud)}/{n}  ({pct:.1f}%)")
        if errors: print(f"errors : {errors} used heuristic fallback")
        print(f"{'─'*55}")
        return fraud


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python run.py <dataset> [eval]")
        print("       dataset: truman | deus | brave")
        sys.exit(1)
    run(sys.argv[1], eval_mode="eval" in sys.argv)
