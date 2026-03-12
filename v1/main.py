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
# Model cascade — order matters: preferred → last resort
# ---------------------------------------------------------------------------
MODELS = [
    "qwen/qwen3-30b-a3b-thinking-2507",   # primary — strongest reasoning
    "qwen/qwen3-14b",                      # fallback — balanced size/quality
    "microsoft/phi-4",                     # last resort — lightweight
]


class ChallengeSystem:
    """
    Three-tier dispatch for fraud detection:
      Zero-LLM zone  (risk ≤ ZERO_LO or ≥ ZERO_HI):  pure heuristic, 0 API calls
      Fast path      (risk ≤ FAST_LOW or ≥ FAST_HIGH): 1 LLM call
      Coop path      (risk in uncertain zone):          2 independent calls + 1 decision
    """

    # Zero-LLM zone: pure heuristic, no API call at all (saves tokens)
    ZERO_LO   = 1    # risk ≤ this  → definitely legit   → return 0 (no call)
    ZERO_HI   = 18   # risk ≥ this  → definitely fraud   → return 1 (no call)
    # Fast-LLM zone: 1 call
    FAST_LOW  = 3    # risk ≤ this  → likely legit
    FAST_HIGH = 15   # risk ≥ this  → likely fraud
    # Cooperative zone: 3 calls (best quality, for ambiguous cases)

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
            temperature= 0.1,   # low temperature: more deterministic fraud decisions
            max_tokens = 512,
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

    # ------------------------------------------------------------------
    # Agent A — TransactionAgent  (independent, parallel-safe)
    # ------------------------------------------------------------------

    def transaction_agent(
        self,
        session_id   : str,
        features     : dict,
        pop_stats    : dict = None,
    ) -> str:
        """
        Assesses the transaction itself: amount anomaly, timing, balance impact.
        Does NOT see sender profile or communications — fully independent lens.
        """
        pop_ctx = ""
        if pop_stats:
            pop_ctx = (
                f"Population context: mean amount={pop_stats['amount_mean']:.2f}, "
                f"std={pop_stats['amount_std']:.2f}, "
                f"night-transaction rate={pop_stats['night_rate']:.2%}.\n"
            )
        system = (
            "You are a financial transaction anomaly analyst for MirrorPay.\n"
            "Analyze the raw transaction metrics provided. Write 2–3 sentences identifying "
            "any red flags: unusually large amounts, suspicious timing (late night, weekend), "
            "negative balance after transaction, new / unknown recipient, or unusual payment method.\n"
            "Be specific and quantitative. Do NOT make a final fraud decision."
        )
        user = (
            f"{pop_ctx}"
            f"Transaction type : {features.get('tx_type', 'N/A')}\n"
            f"Amount           : {features.get('amount', 0):.2f}\n"
            f"Amount z-score vs sender history : {features.get('amount_zscore', 0):.2f}\n"
            f"  (sender avg={features.get('sender_amount_mean', 0):.2f}, "
            f"std={features.get('sender_amount_std', 0):.2f})\n"
            f"Balance after tx : {features.get('balance_after', 0):.2f} "
            f"(negative: {features.get('balance_negative', 0)})\n"
            f"Balance spent pct: {features.get('balance_ratio', 0):.1%} of account\n"
            f"Hour of day      : {features.get('hour', 0):02d}:xx  "
            f"(night flag: {features.get('is_night', 0)}, "
            f"weekend: {features.get('is_weekend', 0)})\n"
            f"Payment method   : {features.get('payment_method', 'N/A')}\n"
            f"Method is unusual for sender: {features.get('unusual_method', 0)}\n"
            f"Recipient new (never seen before): {features.get('recipient_new', 0)}\n"
            f"Sender tx count  : {features.get('sender_tx_count', 0)}\n"
            f"Type is unusual for sender: {features.get('unusual_type', 0)}\n\n"
            "Transaction anomaly assessment (2–3 sentences):"
        )
        return self.call_with_fallback(session_id, system, user, "TransactionAgent")

    # ------------------------------------------------------------------
    # Agent B — ContextAgent  (independent, parallel-safe)
    # ------------------------------------------------------------------

    def context_agent(
        self,
        session_id   : str,
        features     : dict,
    ) -> str:
        """
        Assesses sender context: user profile, GPS consistency, communication signals.
        Does NOT see transaction amount or timing — fully independent lens.
        """
        system = (
            "You are a behavioural and contextual fraud analyst for MirrorPay.\n"
            "Analyze the sender's profile, geographic footprint, and communication patterns.\n"
            "Write 2–3 sentences identifying red flags: age / risk profile, GPS location "
            "inconsistency with transaction site, or phishing/social-engineering signals "
            "in SMS or e-mail communications.\n"
            "Be specific and factual. Do NOT make a final fraud decision."
        )
        user = (
            f"Sender age              : {features.get('sender_age', 'N/A')}\n"
            f"GPS match at tx time    : {features.get('gps_match', 'N/A')}  "
            f"(distance km: {features.get('gps_distance_km', 'N/A')})\n"
            f"Has GPS data near tx    : {features.get('has_gps_data', 0)}\n"
            f"Sender known location   : {features.get('sender_city', 'N/A')}\n"
            f"Transaction location    : {features.get('tx_location', 'N/A')}\n"
            f"Phishing signals in SMS   : {features.get('phishing_sms', 0)}/3\n"
            f"Phishing signals in email : {features.get('phishing_email', 0)}/3\n"
            f"SMS snippet (first 300 chars): {features.get('sms_snippet', '')[:300]}\n"
            f"Email snippet (first 300 chars): {features.get('email_snippet', '')[:300]}\n\n"
            "Contextual risk assessment (2–3 sentences):"
        )
        return self.call_with_fallback(session_id, system, user, "ContextAgent")

    # ------------------------------------------------------------------
    # Agent C — DecisionAgent  (always sync, after A+B or alone)
    # ------------------------------------------------------------------

    def decision_agent(
        self,
        session_id      : str,
        risk_score      : int,
        tx_analysis     : str = "",
        context_analysis: str = "",
        pop_context     : str = "",
    ) -> int:
        """
        Final binary decision: 1 = fraudulent, 0 = legitimate.
        Calibrated toward recall (FN more costly than FP per Rules).
        """
        system = (
            "You are the final fraud decision agent for MirrorPay — Reply Mirror's financial system.\n"
            "Output ONLY the digit '1' or '0' — absolutely nothing else.\n"
            "  1 = this transaction is FRAUDULENT — flag it\n"
            "  0 = this transaction is LEGITIMATE — allow it\n\n"
            "Scoring context (from system rules):\n"
            "  • Missing a real fraud (false negative) causes significant financial damage.\n"
            "  • Blocking a legitimate transaction (false positive) causes reputational loss.\n"
            "  → When evidence is ambiguous or uncertain, output 1 (flag it).\n"
            "  → Output 0 ONLY when legitimacy is explicitly and clearly supported.\n"
            "  → Never flag ALL transactions — only those with real evidence of fraud.\n"
            "  → A risk score ≥ 8 combined with any red flag is sufficient to output 1."
        )
        parts = []
        if pop_context:
            parts.append(pop_context)
        if tx_analysis:
            parts.append(f"Transaction analysis:\n{tx_analysis}")
        if context_analysis:
            parts.append(f"Contextual analysis:\n{context_analysis}")
        parts.append(f"Heuristic risk score: {risk_score}/20")
        parts.append("Decision (output only 1 or 0):")

        user = "\n\n".join(parts)
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
        session_id   : str,
        features     : dict,
        risk_score   : int,
        pop_stats    : dict = None,
        pop_context  : str  = "",
    ) -> int:
        """
        Dispatch:
          risk ≤ ZERO_LO   → return 0 immediately (no LLM call)
          risk ≥ ZERO_HI   → return 1 immediately (no LLM call)
          risk in fast zone → 1 LLM call
          otherwise        → cooperative path (3 LLM calls)

        Returns 0 (legitimate) or 1 (fraudulent).
        """
        # Zero-LLM zone — pure heuristic, saves tokens
        if risk_score <= self.ZERO_LO:
            return 0
        if risk_score >= self.ZERO_HI:
            return 1

        # Fast path — 1 call
        if risk_score <= self.FAST_LOW or risk_score >= self.FAST_HIGH:
            return self.decision_agent(
                session_id, risk_score, pop_context=pop_context
            )

        # Cooperative path — two independent analyses, then decision
        tx_analysis      = self.transaction_agent(session_id, features, pop_stats)
        context_analysis = self.context_agent(session_id, features)

        return self.decision_agent(
            session_id, risk_score,
            tx_analysis      = tx_analysis,
            context_analysis = context_analysis,
            pop_context      = pop_context,
        )


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
