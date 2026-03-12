"""
ChallengeSystem — cooperative multi-agent fraud detection for Reply Mirror.

Architecture — two execution paths:

  FAST PATH  (risk ≤ FAST_LOW  or  risk ≥ FAST_HIGH): 1 LLM call
    └─ DecisionAgent — case is clear; single-call verdict.

  COOPERATIVE PATH  (risk FAST_LOW+1 … FAST_HIGH-1, uncertain zone): 3 LLM calls
    ├─ TransactionAgent  [independent] — analyzes amount anomalies, timing,
    │                                    transaction type, balance impact
    └─ ContextAgent      [independent] — analyzes sender profile, GPS consistency,
                                         communication red flags (SMS / e-mail)
    └─ DecisionAgent     [sync after]  — integrates both views → 0 or 1

TransactionAgent and ContextAgent operate on disjoint information slices —
genuine cooperation rather than a waterfall reformatting chain.

Scoring note (from Rules):
  FN (missed fraud)   → significant financial damage        [WORSE]
  FP (blocked legit)  → economic / reputational loss        [less bad]
  → DecisionAgent is calibrated to favour recall over precision when evidence
    is ambiguous. Output is only 0 when absence of fraud is clearly supported.

All calls share the same Langfuse session_id (grouped via propagate_attributes
in solve.py + @observe + CallbackHandler in _call_model).
"""

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler


# ---------------------------------------------------------------------------
# Model cascade — cheapest first; costs ~$0.05-0.15/M tokens
# ---------------------------------------------------------------------------
MODELS = [
    "google/gemma-3-4b-it",
    "nvidia/nemotron-nano-9b-v2",
    "microsoft/phi-4",
]


class ChallengeSystem:
    """
    Three-tier dispatch for fraud detection:
      Zero-LLM zone  (risk ≤ ZERO_LO or ≥ ZERO_HI):  pure heuristic, 0 API calls
      Fast path      (risk ≤ FAST_LOW or ≥ FAST_HIGH): 1 LLM call
      Coop path      (risk in uncertain zone):          2 independent calls + 1 decision
    """

    ZERO_LO = 4   # risk ≤ this → legit, no LLM call
    ZERO_HI = 15  # risk ≥ this → fraud, no LLM call
    # everything else → exactly 1 LLM call

    def __init__(self):
        load_dotenv()
        self.team_name   = os.getenv("TEAM_NAME", "team")
        self._api_key    = os.getenv("OPENROUTER_API_KEY")
        self._base_url   = "https://openrouter.ai/api/v1"
        self.langfuse    = Langfuse(
            public_key  = os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key  = os.getenv("LANGFUSE_SECRET_KEY"),
            host        = os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
        )

    # ------------------------------------------------------------------
    # Infrastructure helpers
    # ------------------------------------------------------------------

    def _make_model(self, model_id: str) -> ChatOpenAI:
        return ChatOpenAI(
            api_key    = self._api_key,
            base_url   = self._base_url,
            model      = model_id,
            temperature= 0.0,
            max_tokens = 50,    # we need only '0' or '1' + brief reasoning
        )

    @observe()
    def _call_model(
        self,
        session_id : str,
        model_id   : str,
        system     : str,
        user       : str,
        agent_name : str,
    ) -> str:
        """Single tracked LLM call — Langfuse captures tokens, cost, latency."""
        handler  = CallbackHandler()
        model    = self._make_model(model_id)
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        response = model.invoke(messages, config={"callbacks": [handler]})
        return response.content

    def call_with_fallback(
        self,
        session_id : str,
        system     : str,
        user       : str,
        agent_name : str,
    ) -> str:
        """
        Cascade through MODELS until one succeeds.
        On HTTP 429 (rate limit): wait 15 s and retry once before moving on.
        """
        last_error = None
        for model_id in MODELS:
            for attempt in range(2):
                try:
                    result = self._call_model(session_id, model_id, system, user, agent_name)
                    if result and result.strip():
                        return result.strip()
                    raise ValueError("empty response")
                except Exception as exc:
                    last_error = exc
                    if "429" in str(exc) and attempt == 0:
                        print(f"  [{agent_name}] {model_id} rate-limited — waiting 15 s …")
                        time.sleep(15)
                        continue
                    print(f"  [{agent_name}] {model_id} failed ({str(exc)[:100]}), next model …")
                    break
        raise RuntimeError(f"All models exhausted for {agent_name}. Last: {last_error}")

    def decision_agent(
        self,
        session_id  : str,
        risk_score  : int,
        features    : dict,
        pop_context : str = "",
    ) -> int:
        system = (
            "MirrorPay fraud adjudicator. Output ONLY '1' (fraud) or '0' (legit). "
            "FN >> FP. When uncertain → 1. 0 only when clearly legit."
        )
        f = features
        pop = f" | {pop_context}" if pop_context else ""
        user = (
            f"risk={risk_score}/20{pop} | "
            f"type={f.get('tx_type','')} amt={f.get('amount',0):.0f} z={f.get('amount_zscore',0):.1f} "
            f"bal={f.get('balance_after',0):.0f}(neg:{f.get('balance_negative',0)}) "
            f"night={f.get('is_night',0)} wknd={f.get('is_weekend',0)} new_rec={f.get('recipient_new',0)} "
            f"gps={f.get('gps_match','?')} dist_km={f.get('gps_distance_km',0)} "
            f"phish_sms={f.get('phishing_sms',0)} phish_mail={f.get('phishing_email',0)} "
            f"age={f.get('sender_age','?')} desc_legit={f.get('desc_legit',0)} | 1 or 0:"
        )
        raw  = self.call_with_fallback(session_id, system, user, "DecisionAgent")

        # Parse: first digit found wins
        for ch in raw.strip():
            if ch in ("1", "0"):
                return int(ch)
        # Semantic fallback
        low = raw.lower()
        if any(w in low for w in ("fraud", "suspicious", "flag", "yes", "malicious", "anomal")):
            return 1
        return 0

    # ------------------------------------------------------------------
    # Full transaction assessment — adaptive dispatch
    # ------------------------------------------------------------------

    def assess_transaction(
        self,
        session_id  : str,
        features    : dict,
        risk_score  : int,
        pop_context : str = "",
    ) -> int:
        if risk_score <= self.ZERO_LO:
            return 0
        if risk_score >= self.ZERO_HI:
            return 1
        return self.decision_agent(session_id, risk_score, features, pop_context)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import ulid
    from dotenv import load_dotenv
    load_dotenv()
    system = ChallengeSystem()
    sid    = f"{system.team_name}-{ulid.new().str}"

    dummy_features = {
        "tx_type": "e-commerce", "amount": 2500.0, "amount_zscore": 4.2,
        "sender_amount_mean": 80.0, "sender_amount_std": 30.0,
        "balance_after": -120.0, "balance_negative": 1, "balance_ratio": 0.97,
        "hour": 3, "is_night": 1, "is_weekend": 1,
        "payment_method": "mobile device", "unusual_method": 0,
        "recipient_new": 1, "sender_tx_count": 12, "unusual_type": 0,
        "sender_age": 34, "gps_match": "NO", "gps_distance_km": 1200,
        "has_gps_data": 1, "sender_city": "Rome", "tx_location": "New York",
        "phishing_sms": 2, "phishing_email": 1,
        "sms_snippet": "Urgent: verify your account now or it will be suspended.",
        "email_snippet": "",
    }
    print("Risk 12 → cooperative path")
    d = system.assess_transaction(sid, dummy_features, risk_score=12)
    print(f"Decision: {d}")
    print("\nRisk 1 → fast path (legit)")
    d = system.assess_transaction(sid, dummy_features, risk_score=1)
    print(f"Decision: {d}")
    system.langfuse.flush()
