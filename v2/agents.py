"""
Three-agent pipeline:
  ReasonerAgent  — full holistic analysis of everything available
  ScepticAgent   — challenges the reasoner's conclusion (devil's advocate)
  VerdictAgent   — reads both sides, outputs 0 or 1

Fast path: risk <= LOW → 0 direct,  risk >= HIGH → 1 direct (1 call)
"""

import os, time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

load_dotenv()

MODELS = [
    "google/gemma-3-4b-it",
    "nvidia/nemotron-nano-9b-v2",
    "qwen/qwen3-14b:no-thinking",
]

ZERO_LO  = 4   # pure heuristic 0 — no LLM
ZERO_HI  = 15  # pure heuristic 1 — no LLM
# everything else: exactly 1 LLM call — cooperative path eliminated


def _llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        api_key  = os.getenv("OPENROUTER_API_KEY"),
        base_url = "https://openrouter.ai/api/v1",
        model    = model,
        temperature = 0.0,
        max_tokens  = 50,    # need only '0' or '1'
    )


@observe()
def _call(session_id: str, model: str, system: str, user: str, tag: str) -> str:
    h = CallbackHandler()
    r = _llm(model).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)],
        config={"callbacks": [h]},
    )
    return r.content


def _run(session_id: str, system: str, user: str, tag: str) -> str:
    for model in MODELS:
        for attempt in range(2):
            try:
                out = _call(session_id, model, system, user, tag)
                if out and out.strip():
                    return out.strip()
                raise ValueError("empty")
            except Exception as e:
                if "429" in str(e) and attempt == 0:
                    print(f"  [{tag}] rate-limit on {model}, wait 15s…")
                    time.sleep(15)
                    continue
                print(f"  [{tag}] {model} failed: {str(e)[:80]}")
                break
    raise RuntimeError(f"all models failed for {tag}")


# ── Agents ───────────────────────────────────────────────────────────────────

def fast_verdict(session_id: str, risk: int, features: dict) -> int:
    sys = (
        "MirrorPay adjudicator. Output ONLY '1' (fraud) or '0' (legit). "
        "FN>>FP. When uncertain → 1. 0 only if clearly legit."
    )
    f = features
    usr = (
        f"risk={risk}/20 | "
        f"type={f.get('tx_type','')} amt={f.get('amount',0):.0f} z={f.get('amount_zscore',0):.1f} "
        f"bal={f.get('balance_after',0):.0f}(neg:{f.get('balance_negative',0)}) "
        f"night={f.get('is_night',0)} wknd={f.get('is_weekend',0)} new_rec={f.get('recipient_new',0)} "
        f"gps={f.get('gps_match','?')} dist={f.get('gps_distance_km',0)}km "
        f"phish_sms={f.get('phishing_sms',0)} phish_mail={f.get('phishing_email',0)} "
        f"age={f.get('sender_age','?')} desc_legit={f.get('desc_legit',0)} "
        f"| 1 or 0:"
    )
    raw = _run(session_id, sys, usr, "FastVerdict")
    for ch in raw.strip():
        if ch in ("1","0"):
            return int(ch)
    return 1 if risk > 9 else 0


def assess(session_id: str, features: dict, risk: int) -> int:
    if risk <= ZERO_LO:
        return 0
    if risk >= ZERO_HI:
        return 1
    return fast_verdict(session_id, risk, features)


# ── Langfuse client (module-level) ────────────────────────────────────────────

lf = Langfuse(
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key = os.getenv("LANGFUSE_SECRET_KEY"),
    host       = os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
)
