"""
DataAgent — loads and pre-processes all datasets for the Reply Mirror fraud challenge.

Dataset availability grows with challenge level:
  Level 1+: transactions.csv, users.json, locations.json
  Level 2+: conversations.json  (SMS threads per user)
  Level 3+: messages.json       (e-mail threads per user)
  Level 4+: additional data — loaded gracefully if present

CSV column layout (inferred from Rules + example data):
  TransactionID, SenderID, RecipientID, TransactionType, Amount,
  Location, PaymentMethod, SenderIBAN, RecipientIBAN, Balance, Description, Timestamp
  (Description contains free-text transaction notes, e.g. "Salary payment Jan")
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


# Expected columns — the 11th slot ("Description") can be a label in training or a text description
_TX_COLS = [
    "TransactionID", "SenderID", "RecipientID", "TransactionType",
    "Amount", "Location", "PaymentMethod", "SenderIBAN", "RecipientIBAN",
    "Balance", "Description", "Timestamp",
]

# Phishing / social-engineering keywords for SMS / e-mail analysis
_PHISHING_KEYWORDS = [
    "urgent", "verify", "suspend", "click", "confirm", "account",
    "blocked", "wire", "transfer now", "immediately", "security alert",
    "limited time", "unusual activity", "sign in", "reset password",
    "won", "prize", "bitcoin", "crypto", "investment",
]


class DataAgent:
    """Loads, validates, and serves all data sources for the fraud challenge."""

    def __init__(self, data_dir: str = "The+Truman+Show_train/The Truman Show_train/public"):
        self.data_dir = Path(data_dir)
        self.transactions_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        self.locations_df: Optional[pd.DataFrame] = None
        self.conversations: Dict[str, str] = {}   # user_id → SMS text
        self.messages: Dict[str, str] = {}        # user_id → email text
        # Derived caches built lazily
        self._sender_profiles: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all_data(self) -> Dict:
        """Load every available dataset. Missing optional files are skipped."""
        print("=== Loading data ===")
        self.transactions_df = self._load_transactions()
        self.users_df        = self._load_users()
        self.locations_df    = self._load_locations()
        self.conversations   = self._load_conversations()
        self.messages        = self._load_messages()
        print("=== Data ready ===\n")
        return {
            "transactions": self.transactions_df,
            "users":        self.users_df,
            "locations":    self.locations_df,
            "conversations": self.conversations,
            "messages":     self.messages,
        }

    def get_transaction_data(self, tx_id: str) -> Dict:
        """
        Return all context needed to assess one transaction:
          tx        — the transaction row (Series)
          sender    — user profile (Series or None)
          tx_history — all transactions by the same sender (DataFrame)
          gps_near  — sender GPS points within ±2 h of this transaction (DataFrame)
          sms       — SMS thread for sender (str, may be empty)
          email     — e-mail thread for sender (str, may be empty)
        """
        tx = self.transactions_df[
            self.transactions_df["TransactionID"] == tx_id
        ].iloc[0]

        sender_id = tx["SenderID"]

        # Sender user profile
        sender = None
        if self.users_df is not None and len(self.users_df) > 0 and "iban" in self.users_df.columns:
            sender_iban = str(tx.get("SenderIBAN") or "")
            if sender_iban:
                rows = self.users_df[self.users_df["iban"] == sender_iban]
                if len(rows) > 0:
                    sender = rows.iloc[0]

        # Full transaction history for this sender (sorted ascending)
        tx_history = self.transactions_df[
            self.transactions_df["SenderID"] == sender_id
        ].copy()

        # GPS points near the transaction timestamp (±2 hours)
        gps_near = pd.DataFrame()
        if self.locations_df is not None and len(self.locations_df) > 0:
            ts = tx["Timestamp"]
            window = pd.Timedelta(hours=2)
            mask = (
                (self.locations_df["user_id"] == sender_id) &
                (self.locations_df["timestamp"] >= ts - window) &
                (self.locations_df["timestamp"] <= ts + window)
            )
            gps_near = self.locations_df[mask].copy()

        return {
            "tx":          tx,
            "sender":      sender,
            "tx_history":  tx_history,
            "gps_near":    gps_near,
            "sms":         self.conversations.get(str(tx.get("SenderIBAN") or ""), ""),
            "email":       self.messages.get(str(tx.get("SenderIBAN") or ""), ""),
        }

    def build_sender_profiles(self) -> Dict[str, Dict]:
        """
        Pre-compute per-sender behavioral statistics over the full transaction set.
        Cached after first call.

        Returns:
            sender_id → {
              tx_count, amount_mean, amount_std, amount_max,
              common_type, common_method, night_rate,
              known_recipients (set), phishing_score (0-3)
            }
        """
        if self._sender_profiles is not None:
            return self._sender_profiles

        profiles: Dict[str, Dict] = {}
        for sender_id, grp in self.transactions_df.groupby("SenderID"):
            amounts = grp["Amount"]
            night_mask = grp["Timestamp"].dt.hour.between(0, 5)
            # look up communications by SenderIBAN (dicts are iban-keyed)
            iban_vals = grp["SenderIBAN"].dropna().replace("", None).dropna()
            iban_key  = str(iban_vals.iloc[0]) if len(iban_vals) > 0 else ""
            combined_text = self.conversations.get(iban_key, "") + " " + self.messages.get(iban_key, "")
            profiles[sender_id] = {
                "tx_count":          len(grp),
                "amount_mean":       float(amounts.mean()),
                "amount_std":        float(amounts.std(ddof=0)) if len(grp) > 1 else 0.0,
                "amount_max":        float(amounts.max()),
                "common_type":       grp["TransactionType"].mode().iloc[0] if len(grp) > 0 and len(grp["TransactionType"].mode()) > 0 else "",
                "common_method":     grp["PaymentMethod"].mode().iloc[0]   if len(grp) > 0 and len(grp["PaymentMethod"].mode()) > 0 else "",
                "night_rate":        float(night_mask.mean()),
                "known_recipients":  set(grp["RecipientID"].dropna().unique()),
                "phishing_score":    self._phishing_score(combined_text),
            }

        self._sender_profiles = profiles
        return profiles

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_transactions(self) -> pd.DataFrame:
        path = self.data_dir / "transactions.csv"
        # Try reading with header first; fall back to manual column assignment
        df = pd.read_csv(path, header=None, dtype=str)
        # If first row looks like a header (non-UUID text), drop it
        first = str(df.iloc[0, 0])
        if not _looks_like_uuid(first) and not _looks_like_id(first):
            df = df.iloc[1:].reset_index(drop=True)

        # Assign columns robustly
        if df.shape[1] >= len(_TX_COLS):
            df.columns = list(_TX_COLS) + [f"_extra_{i}" for i in range(df.shape[1] - len(_TX_COLS))]
        else:
            df.columns = _TX_COLS[: df.shape[1]]

        # Type conversions
        df["Amount"]    = pd.to_numeric(df["Amount"],  errors="coerce").fillna(0.0)
        df["Balance"]   = pd.to_numeric(df["Balance"], errors="coerce").fillna(0.0)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values("Timestamp").reset_index(drop=True)

        # Derived time columns (used heavily in feature extraction)
        df["_hour"]    = df["Timestamp"].dt.hour
        df["_weekday"] = df["Timestamp"].dt.weekday  # 0=Mon … 6=Sun
        df["_is_night"]   = df["_hour"].between(0, 5).astype(int)
        df["_is_weekend"] = (df["_weekday"] >= 5).astype(int)

        n_tx = len(df)
        n_senders = df["SenderID"].nunique()
        print(f"  transactions.csv : {n_tx} transactions, {n_senders} unique senders")
        return df

    def _load_users(self) -> Optional[pd.DataFrame]:
        path = self.data_dir / "users.json"
        if not path.exists():
            print("  users.json       : not found, skipping")
            return pd.DataFrame()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if "birth_year" in df.columns:
            df["age"] = 2087 - df["birth_year"]   # challenge year is 2087
        if "residence" in df.columns:
            df["residence_lat"]  = df["residence"].apply(lambda x: float(x["lat"])  if isinstance(x, dict) else None)
            df["residence_lng"]  = df["residence"].apply(lambda x: float(x["lng"])  if isinstance(x, dict) else None)
            df["residence_city"] = df["residence"].apply(lambda x: x.get("city", "") if isinstance(x, dict) else "")
            df.drop(columns=["residence"], inplace=True)
        print(f"  users.json       : {len(df)} users")
        return df

    def _load_locations(self) -> pd.DataFrame:
        path = self.data_dir / "locations.json"
        if not path.exists():
            print("  locations.json   : not found, skipping")
            return pd.DataFrame()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # Normalise column names across levels
        df.rename(columns={
            "BioTag": "user_id", "biotag": "user_id",
            "Datetime": "timestamp", "datetime": "timestamp",
            "Lat": "lat", "Lng": "lng",
        }, inplace=True)
        df["lat"]       = pd.to_numeric(df["lat"],  errors="coerce")
        df["lng"]       = pd.to_numeric(df["lng"],  errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        print(f"  locations.json   : {len(df)} GPS points, {df['user_id'].nunique()} users")
        return df

    def _load_conversations(self) -> Dict[str, str]:
        """Load SMS threads. Returns iban -> concatenated SMS text.
        Since sms.json has no user ID field, we parse the first name from
        the 'Hi <name>' greeting and match against users by first_name.
        Key is users.iban so it can be looked up via tx.SenderIBAN.
        """
        for fname in ("sms.json", "conversations.json", "conversations.csv"):
            path = self.data_dir / fname
            if path.exists():
                break
        else:
            print("  sms              : not found, skipping")
            return {}

        with open(path, encoding="utf-8") as f:
            if fname.endswith(".json"):
                data = json.load(f)
            else:
                data = json.loads(pd.read_csv(path).to_json(orient="records"))

        # Build first_name (lower) -> iban index from users
        fname_iban: Dict[str, str] = {}
        if self.users_df is not None and len(self.users_df) > 0:
            for _, u in self.users_df.iterrows():
                fn = str(u.get("first_name") or "").strip().lower()
                if fn and "iban" in u.index:
                    fname_iban[fn] = str(u["iban"])

        result: Dict[str, str] = {}
        for record in (data if isinstance(data, list) else [data]):
            # Try explicit ID fields first (future-proof)
            uid = str(record.get("UserID") or record.get("user_id") or record.get("BioTag") or "")
            text = str(record.get("SMS") or record.get("sms") or record.get("text") or record.get("content") or "")
            # If no uid, parse 'Hi FirstName' from message body
            if not uid and fname_iban and text:
                m = re.search(r'(?:Hi|Hello|Dear)\s+([A-Z][a-z]+)', text)
                if m:
                    uid = fname_iban.get(m.group(1).lower(), "")
            if uid:
                result[uid] = result.get(uid, "") + " " + text

        print(f"  sms              : {len(result)} threads linked")
        return result

    def _load_messages(self) -> Dict[str, str]:
        """Load e-mail threads. Returns iban -> concatenated mail text.
        Parses 'To: "FirstName LastName"' from mail headers to find the user.
        Key is users.iban so it can be looked up via tx.SenderIBAN.
        """
        for fname in ("mails.json", "messages.json", "messages.csv"):
            path = self.data_dir / fname
            if path.exists():
                break
        else:
            print("  mails            : not found, skipping")
            return {}

        with open(path, encoding="utf-8") as f:
            if fname.endswith(".json"):
                data = json.load(f)
            else:
                data = json.loads(pd.read_csv(path).to_json(orient="records"))

        # Build fullname (lower) -> iban index from users
        fullname_iban: Dict[str, str] = {}
        if self.users_df is not None and len(self.users_df) > 0:
            for _, u in self.users_df.iterrows():
                fn = str(u.get("first_name") or "").strip()
                ln = str(u.get("last_name")  or "").strip()
                full = (fn + " " + ln).strip().lower()
                if full and "iban" in u.index:
                    fullname_iban[full] = str(u["iban"])

        result: Dict[str, str] = {}
        for record in (data if isinstance(data, list) else [data]):
            uid  = str(record.get("UserID") or record.get("user_id") or record.get("BioTag") or "")
            mail = str(record.get("mail") or record.get("Mail") or record.get("email") or record.get("text") or record.get("content") or "")
            # If no uid, extract 'To: "FirstName LastName"' from headers
            if not uid and fullname_iban and mail:
                m = re.search(r'To:\s*"?([A-Z][a-z]+\s+[A-Z][a-z]+)"?', mail)
                if m:
                    uid = fullname_iban.get(m.group(1).lower(), "")
            if uid:
                result[uid] = result.get(uid, "") + " " + mail

        print(f"  mails            : {len(result)} threads linked")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _phishing_score(text: str) -> int:
        """Count how many phishing keyword families appear in text (capped at 3)."""
        lower = text.lower()
        return min(3, sum(1 for kw in _PHISHING_KEYWORDS if kw in lower))


# ------------------------------------------------------------------
# Tiny helpers used during CSV loading
# ------------------------------------------------------------------

def _looks_like_uuid(s: str) -> bool:
    return bool(re.match(r"[0-9a-f]{8}-[0-9a-f]{4}-", s, re.IGNORECASE))

def _looks_like_id(s: str) -> bool:
    """Heuristic: typical ID format like SCHV-SVRA-7BC-COR-0."""
    return bool(re.match(r"[A-Z0-9]{1,6}-", s))


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "truman"
    from solve import _resolve_dataset_path
    path = _resolve_dataset_path(name, False)
    agent = DataAgent(path)
    data  = agent.load_all_data()
    tx    = data["transactions"]
    print(f"\nTransactions : {len(tx)}")
    print(f"Types        : {tx['TransactionType'].value_counts().to_dict()}")
    print(f"Amount range : {tx['Amount'].min():.2f} – {tx['Amount'].max():.2f}")
    print(f"Date range   : {tx['Timestamp'].min()} – {tx['Timestamp'].max()}")
    profiles = agent.build_sender_profiles()
    print(f"Sender profiles: {len(profiles)}")
