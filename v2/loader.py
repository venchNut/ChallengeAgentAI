import json, re, csv
import pandas as pd
from pathlib import Path

YEAR = 2087

PHISH = [
    "urgent","verify","suspend","click","confirm","blocked","wire",
    "transfer now","immediately","security alert","limited time",
    "unusual activity","sign in","reset password","won","prize",
    "bitcoin","crypto","investment",
]

_DATASETS = {
    "truman": "The+Truman+Show",
    "deus":   "Deus+Ex",
    "brave":  "Brave+New+World",
}

_TX_COLS = [
    "TransactionID","SenderID","RecipientID","TransactionType",
    "Amount","Location","PaymentMethod","SenderIBAN","RecipientIBAN",
    "Balance","Label","Timestamp",
]


def dataset_path(name: str, eval_mode: bool) -> Path:
    key = name.lower()
    if key not in _DATASETS:
        raise ValueError(f"unknown dataset '{name}' — use: truman, deus, brave")
    folder = _DATASETS[key]
    suffix = "eval" if eval_mode else "train"
    plain  = folder.replace("+", " ")
    return Path(f"{folder}_{suffix}/{plain}_{suffix}/public")


def phish_score(text: str) -> int:
    lo = text.lower()
    return min(3, sum(1 for k in PHISH if k in lo))


def load(name: str, eval_mode: bool = False) -> dict:
    root = dataset_path(name, eval_mode)
    print(f"loading {root}")

    # transactions
    raw = pd.read_csv(root / "transactions.csv", header=None, dtype=str)
    first = str(raw.iloc[0, 0])
    if not re.match(r"[A-Z0-9]{1,8}-", first) and not re.match(r"[0-9a-f]{8}-", first, re.I):
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.shape[1] >= len(_TX_COLS):
        raw.columns = list(_TX_COLS) + [f"_x{i}" for i in range(raw.shape[1] - len(_TX_COLS))]
    else:
        raw.columns = _TX_COLS[:raw.shape[1]]
    raw["Amount"]    = pd.to_numeric(raw["Amount"],  errors="coerce").fillna(0.0)
    raw["Balance"]   = pd.to_numeric(raw["Balance"], errors="coerce").fillna(0.0)
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
    raw = raw.sort_values("Timestamp").reset_index(drop=True)
    raw["_h"]  = raw["Timestamp"].dt.hour
    raw["_wd"] = raw["Timestamp"].dt.weekday
    raw["_ni"] = raw["_h"].between(0, 5).astype(int)
    raw["_we"] = (raw["_wd"] >= 5).astype(int)
    print(f"  tx: {len(raw)}  senders: {raw['SenderID'].nunique()}")

    # users
    users = pd.DataFrame()
    p = root / "users.json"
    if p.exists():
        users = pd.DataFrame(json.load(open(p, encoding="utf-8")))
        if "birth_year" in users.columns:
            users["age"] = YEAR - users["birth_year"]
        if "residence" in users.columns:
            users["res_lat"]  = users["residence"].apply(lambda x: float(x["lat"])  if isinstance(x, dict) else None)
            users["res_lng"]  = users["residence"].apply(lambda x: float(x["lng"])  if isinstance(x, dict) else None)
            users["res_city"] = users["residence"].apply(lambda x: x.get("city","") if isinstance(x, dict) else "")
            users.drop(columns=["residence"], inplace=True)
        print(f"  users: {len(users)}")

    # locations
    locs = pd.DataFrame()
    p = root / "locations.json"
    if p.exists():
        locs = pd.DataFrame(json.load(open(p, encoding="utf-8")))
        locs.rename(columns={
            "BioTag":"user_id","biotag":"user_id",
            "Datetime":"ts","datetime":"ts",
            "Lat":"lat","Lng":"lng",
        }, inplace=True)
        if "timestamp" in locs.columns and "ts" not in locs.columns:
            locs.rename(columns={"timestamp":"ts"}, inplace=True)
        locs["lat"] = pd.to_numeric(locs["lat"], errors="coerce")
        locs["lng"] = pd.to_numeric(locs["lng"], errors="coerce")
        locs["ts"]  = pd.to_datetime(locs["ts"], errors="coerce")
        locs = locs.sort_values(["user_id","ts"]).reset_index(drop=True)
        print(f"  locs: {len(locs)}")

    # sms
    sms_map: dict = {}
    for fname in ("sms.json","conversations.json"):
        p = root / fname
        if p.exists():
            data = json.load(open(p, encoding="utf-8"))
            for r in (data if isinstance(data, list) else [data]):
                uid = str(r.get("UserID") or r.get("user_id") or r.get("BioTag") or "")
                txt = str(r.get("SMS") or r.get("sms") or r.get("text") or "")
                if uid: sms_map[uid] = sms_map.get(uid,"") + " " + txt
            print(f"  sms: {len(sms_map)}")
            break

    # mails
    mail_map: dict = {}
    for fname in ("mails.json","messages.json"):
        p = root / fname
        if p.exists():
            data = json.load(open(p, encoding="utf-8"))
            for r in (data if isinstance(data, list) else [data]):
                uid = str(r.get("UserID") or r.get("user_id") or r.get("BioTag") or "")
                txt = str(r.get("mail") or r.get("Mail") or r.get("email") or r.get("text") or "")
                if uid: mail_map[uid] = mail_map.get(uid,"") + " " + txt
            print(f"  mails: {len(mail_map)}")
            break

    return dict(tx=raw, users=users, locs=locs, sms=sms_map, mails=mail_map)


def build_profiles(tx: pd.DataFrame, sms: dict, mails: dict) -> dict:
    profiles = {}
    for sid, g in tx.groupby("SenderID"):
        amt = g["Amount"]
        combined_text = sms.get(sid,"") + " " + mails.get(sid,"")
        profiles[sid] = {
            "n":           len(g),
            "amt_mean":    float(amt.mean()),
            "amt_std":     float(amt.std(ddof=0)) if len(g) > 1 else 0.0,
            "amt_max":     float(amt.max()),
            "top_type":    g["TransactionType"].mode().iloc[0] if len(g) > 0 and len(g["TransactionType"].mode()) > 0 else "",
            "top_method":  g["PaymentMethod"].mode().iloc[0]   if len(g) > 0 and len(g["PaymentMethod"].mode()) > 0 else "",
            "night_rate":  float(g["_ni"].mean()),
            "recipients":  set(g["RecipientID"].dropna().unique()),
            "phish":       phish_score(combined_text),
        }
    return profiles


def get_tx_context(tid: str, data: dict) -> dict:
    tx   = data["tx"]
    row  = tx[tx["TransactionID"] == tid].iloc[0]
    sid  = row["SenderID"]
    ts   = row["Timestamp"]

    sender = None
    if len(data["users"]) > 0 and "iban" in data["users"].columns:
        sender_iban = str(row.get("SenderIBAN") or "")
        if sender_iban:
            m = data["users"][data["users"]["iban"] == sender_iban]
            if len(m): sender = m.iloc[0]

    history = tx[tx["SenderID"] == sid].copy()

    gps = pd.DataFrame()
    if len(data["locs"]) > 0:
        win  = pd.Timedelta(hours=2)
        mask = (
            (data["locs"]["user_id"] == sid) &
            (data["locs"]["ts"] >= ts - win) &
            (data["locs"]["ts"] <= ts + win)
        )
        gps = data["locs"][mask].copy()

    return dict(row=row, sender=sender, history=history, gps=gps,
                sms=data["sms"].get(sid,""), mail=data["mails"].get(sid,""))
