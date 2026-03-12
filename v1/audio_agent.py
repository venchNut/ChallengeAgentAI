"""
AudioAgent — transcribes MP3 call recordings and extracts phishing signals.

Filename convention expected by the challenge:
  YYYYMMDD_HHMMSS-firstname_lastname.mp3
  e.g. 20870102_093522-amanda_powell.mp3

Flow:
  1. Scan audio/ folder for .mp3 files
  2. Load transcript cache from  audio_transcripts.json  (same public/ dir)
  3. Transcribe any uncached files with faster-whisper tiny (CPU, int8)
  4. Save updated cache
  5. Build per-IBAN aggregated features (max phishing score, call count, snippet)

Install dep (one-time):
  pip install faster-whisper

Integration example:
  from audio_agent import AudioAgent
  aa = AudioAgent(data_dir, users_raw_list)
  aa.load()
  feat = aa.get_features(iban)   # {'phishing_score': 2, 'call_count': 3, 'snippet': '...', 'calls': [...]}
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Phishing keyword list tailored for phone-call transcripts
# ---------------------------------------------------------------------------
_AUDIO_PHISHING_KW = [
    # urgency / authority
    "urgent", "immediately", "right now", "today only", "expire",
    "last chance", "time is running out",
    # account takeover
    "verify", "confirm", "suspend", "block", "restrict",
    "unusual activity", "suspicious", "unauthorised", "unauthorized",
    "sign in", "log in", "password", "reset", "pin",
    # OTP / code fishing
    "one-time", "otp", "code", "authorization code", "6-digit", "4-digit",
    # financial pressure
    "wire", "transfer", "arrest", "fine", "penalty", "tax",
    "owe", "debt", "refund", "reimburs",
    "bitcoin", "crypto", "investment", "wallet",
    # social engineering openers
    "press 1", "press 2", "call back", "customer service",
    "technical support", "helpdesk",
    # generic fraud
    "prize", "won", "lottery", "claim",
]


def _phishing_score(text: str) -> int:
    """Return 0–5 count of distinct phishing keyword families found in text."""
    lower = text.lower()
    return min(5, sum(1 for kw in _AUDIO_PHISHING_KW if kw in lower))


def _parse_filename(stem: str):
    """
    Parse '20870102_093522-amanda_powell'
    → (datetime(2087,1,2,9,35,22), 'amanda', 'powell')

    Returns (None, '', '') on any parse failure.
    """
    m = re.match(r"(\d{8})_(\d{6})-(.+)$", stem)
    if not m:
        return None, "", ""
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        dt = None
    name_raw   = m.group(3)
    parts      = name_raw.split("_")
    first_name = parts[0]
    last_name  = "_".join(parts[1:]) if len(parts) > 1 else ""
    return dt, first_name, last_name


# ---------------------------------------------------------------------------
# AudioAgent
# ---------------------------------------------------------------------------

class AudioAgent:
    """
    Transcribes MP3 recordings and exposes per-IBAN phishing feature dict.

    Parameters
    ----------
    data_dir : str
        Path to the dataset's public/ folder (must contain an audio/ subfolder).
    users : list
        Raw list of user dicts (from users.json), each with keys
        first_name, last_name, iban.
    """

    def __init__(self, data_dir: str, users: list):
        self.data_dir   = Path(data_dir)
        self.audio_dir  = self.data_dir / "audio"
        self.cache_path = self.data_dir / "audio_transcripts.json"
        self._users     = users
        self._name_iban: Dict[str, str] = {}      # "first|last" → iban
        self._iban_features: Dict[str, dict] = {} # iban → feature dict
        self._build_name_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, dict]:
        """
        Transcribe all MP3s (using cache when possible) and index by IBAN.

        Returns
        -------
        Dict iban → {
            phishing_score : int  (0–5, max across all calls)
            call_count     : int
            snippet        : str  (top-2 suspicious call excerpts)
            calls          : list of { ts, text, phishing_score }
        }
        """
        if not self.audio_dir.exists():
            print(f"  audio            : folder not found ({self.audio_dir}), skipping")
            return {}

        mp3_files = sorted(self.audio_dir.glob("*.mp3"))
        if not mp3_files:
            print("  audio            : 0 MP3 files, skipping")
            return {}
        print(f"  audio            : {len(mp3_files)} MP3 files found")

        # --- Load transcript cache ---
        cache: Dict[str, str] = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, encoding="utf-8") as fh:
                    cache = json.load(fh)
                print(f"  audio cache      : {len(cache)}/{len(mp3_files)} already transcribed")
            except Exception as exc:
                print(f"  audio cache      : corrupt, rebuilding ({exc})")
                cache = {}

        # --- Transcribe missing files ---
        to_do = [p for p in mp3_files if p.name not in cache]
        if to_do:
            model = self._load_whisper_model()
            if model is None:
                print("  audio            : faster-whisper not installed.")
                print("                     Run:  pip install faster-whisper")
                print("                     Continuing with cached transcripts only.")
            else:
                print(f"  audio            : transcribing {len(to_do)} new files …")
                for i, mp3_path in enumerate(to_do, 1):
                    try:
                        text = self._transcribe(model, str(mp3_path))
                        cache[mp3_path.name] = text
                    except Exception as exc:
                        print(f"    WARNING {mp3_path.name}: {exc}")
                        cache[mp3_path.name] = ""

                    # Progress + incremental save every 20 files
                    if i % 20 == 0 or i == len(to_do):
                        print(f"    {i}/{len(to_do)} transcribed")
                        self._save_cache(cache)

                print(f"  audio            : transcription complete, cache → {self.cache_path.name}")

        # --- Build per-IBAN aggregates ---
        iban_calls: Dict[str, List[dict]] = {}
        for mp3_name, text in cache.items():
            stem                  = Path(mp3_name).stem
            dt, first, last       = _parse_filename(stem)
            iban                  = (
                self._name_iban.get(f"{first.lower()}|{last.lower()}")
                or self._name_iban.get(f"{first.lower()}|")   # first-name fallback
            )
            if not iban:
                continue
            phish = _phishing_score(text or "")
            iban_calls.setdefault(iban, []).append({
                "ts":            dt.isoformat() if dt else "",
                "text":          text or "",
                "phishing_score": phish,
            })

        for iban, calls in iban_calls.items():
            calls_sorted = sorted(calls, key=lambda x: -x["phishing_score"])
            snippet = " | ".join(c["text"][:160] for c in calls_sorted[:2])
            self._iban_features[iban] = {
                "phishing_score": max(c["phishing_score"] for c in calls),
                "call_count":     len(calls),
                "snippet":        snippet,
                "calls":          calls,
            }

        n_linked = len(self._iban_features)
        n_calls  = sum(len(v) for v in iban_calls.values())
        print(f"  audio            : {n_calls} calls linked to {n_linked} IBANs")
        return self._iban_features

    def get_features(self, iban: str) -> dict:
        """Return audio feature dict for an IBAN, or empty dict if none."""
        return self._iban_features.get(str(iban), {})

    def has_audio(self) -> bool:
        return bool(self._iban_features)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_name_index(self):
        """Map 'first|last' (lower) → iban for fast filename-based lookup."""
        for u in self._users:
            fn   = str(u.get("first_name") or "").strip().lower()
            ln   = str(u.get("last_name")  or "").strip().lower()
            iban = str(u.get("iban") or "")
            if fn and iban:
                self._name_iban[f"{fn}|{ln}"] = iban   # full match
                self._name_iban[f"{fn}|"]     = iban   # first-name-only fallback

    def _save_cache(self, cache: dict):
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Whisper backend (lazy, CPU-only, tiny model)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_whisper_model():
        """
        Load faster-whisper tiny model (int8 quantised, CPU).
        Downloads ~75 MB on first use (cached by HuggingFace hub).
        Returns model or None if faster-whisper is not installed.
        """
        try:
            from faster_whisper import WhisperModel  # type: ignore
            print("  audio            : loading faster-whisper tiny model …")
            return WhisperModel("tiny", device="cpu", compute_type="int8")
        except ImportError:
            return None

    @staticmethod
    def _transcribe(model, path: str) -> str:
        """Transcribe one audio file, return full text string."""
        segments, _info = model.transcribe(
            path,
            beam_size=1,        # fast greedy decoding
            language=None,      # auto-detect language
            vad_filter=True,    # skip silent segments
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_agent.py <public_dir>")
        print("  e.g. python audio_agent.py ../1984_eval/public")
        sys.exit(1)

    pub_dir = sys.argv[1]
    users_path = Path(pub_dir) / "users.json"
    if not users_path.exists():
        print(f"users.json not found at {users_path}")
        sys.exit(1)

    with open(users_path, encoding="utf-8") as fh:
        users_raw = json.load(fh)

    aa = AudioAgent(pub_dir, users_raw)
    features = aa.load()

    print(f"\nTotal IBANs with audio: {len(features)}")
    for iban, feat in list(features.items())[:5]:
        print(f"  {iban[:20]}…  phish={feat['phishing_score']}  calls={feat['call_count']}")
        print(f"    snippet: {feat['snippet'][:120]}")
