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
    "qwen/qwen3-30b-a3b-thinking-2507",
    "qwen/qwen3-14b",
    "microsoft/phi-4",
]

ZERO_LO  = 1   # pure heuristic 0 — no LLM
ZERO_HI  = 18  # pure heuristic 1 — no LLM
FAST_LO  = 3   # 1-call path
FAST_HI  = 15  # 1-call path


def _llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        api_key  = os.getenv("OPENROUTER_API_KEY"),
        base_url = "https://openrouter.ai/api/v1",
        model    = model,
        temperature = 0.1,
        max_tokens  = 600,
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

def reasoner(session_id: str, features: dict, pop_ctx: str = "") -> str:
    sys = (
        "You are a fraud intelligence analyst at MirrorPay (year 2087).\n"
        "Given all available data about a transaction and its sender, write a "
        "concise analysis (3–5 sentences) identifying ALL signals — both "
        "suspicious and reassuring. Be quantitative and specific. "
        "Do NOT give a final verdict."
    )
    ctx = (f"Population context: {pop_ctx}\n\n" if pop_ctx else "")
    usr = ctx + _features_block(features)
    return _run(session_id, sys, usr, "Reasoner")


def sceptic(session_id: str, features: dict, reasoning: str) -> str:
    sys = (
        "You are a fraud detection sceptic at MirrorPay.\n"
        "You have read a colleague's analysis. Your job is to challenge it: "
        "identify any overlooked signals that CONTRADICT their conclusion, "
        "or amplify signals they downplayed. Write 2–3 sentences. "
        "Do NOT give a final verdict."
    )
    usr = f"Colleague's analysis:\n{reasoning}\n\nRaw data:\n{_features_block(features)}"
    return _run(session_id, sys, usr, "Sceptic")


def verdict(session_id: str, risk: int, reasoning: str, challenge: str) -> int:
    sys = (
        "You are the final fraud adjudicator at MirrorPay.\n"
        "Output ONLY '1' (fraud) or '0' (legitimate). Nothing else.\n"
        "Rules:\n"
        "  • False negative (missed fraud) = large financial damage — avoid.\n"
        "  • False positive (blocked legit) = reputational cost — acceptable.\n"
        "  → Flag (1) whenever evidence is ambiguous or uncertain.\n"
        "  → Output 0 ONLY when legitimacy is explicitly and clearly supported.\n"
        "  → A risk score ≥ 8 with any red flag is sufficient to output 1.\n"
        "  → Never flag ALL transactions."
    )
    usr = (
        f"Heuristic risk score: {risk}/20\n\n"
        f"Analysis:\n{reasoning}\n\n"
        f"Counter-analysis:\n{challenge}\n\n"
        "Decision (1 or 0):"
    )
    raw = _run(session_id, sys, usr, "Verdict")
    for ch in raw.strip():
        if ch in ("1","0"):
            return int(ch)
    lo = raw.lower()
    if any(w in lo for w in ("fraud","suspicious","flag","yes","1")):
        return 1
    return 0


def fast_verdict(session_id: str, risk: int, features: dict) -> int:
    sys = (
        "You are a fraud adjudicator at MirrorPay. "
        "Output ONLY '1' (fraud) or '0' (legitimate). Nothing else.\n"
        "False negatives (missed fraud) are far worse than false positives.\n"
        "When uncertain, output 1. Output 0 only when clearly legit."
    )
    usr = f"Heuristic risk score: {risk}/20 (decisive case — use this as primary signal)\n\n{_features_block(features)}\nDecision (1 or 0):"
    raw = _run(session_id, sys, usr, "FastVerdict")
    for ch in raw.strip():
        if ch in ("1","0"):
            return int(ch)
    return 1 if risk >= FAST_HI else 0


def assess(session_id: str, features: dict, risk: int, pop_ctx: str = "") -> int:
    # Zero-LLM zone — saves tokens on obvious cases
    if risk <= ZERO_LO:
        return 0
    if risk >= ZERO_HI:
        return 1
    # 1-call fast zone
    if risk <= FAST_LO or risk >= FAST_HI:
        return fast_verdict(session_id, risk, features)
    # 3-call cooperative zone
    r = reasoner(session_id, features, pop_ctx)
    c = sceptic(session_id, features, r)
    return verdict(session_id, risk, r, c)


# ── Feature block formatter ───────────────────────────────────────────────────

def _features_block(f: dict) -> str:
    lines = [
        f"TX type        : {f.get('tx_type','')}",
        f"Amount         : {f.get('amount',0):.2f}  (sender avg {f.get('amt_mean',0):.2f} ± {f.get('amt_std',0):.2f}, z={f.get('z',0):.2f})",
        f"Balance after  : {f.get('balance',0):.2f}  (negative: {f.get('neg_bal',0)})",
        f"Time           : {f.get('hour',0):02d}:xx  night={f.get('night',0)}  weekend={f.get('wknd',0)}",
        f"Payment method : {f.get('method','')}  (unusual for sender: {f.get('unusual_method',0)})",
        f"Recipient new  : {f.get('new_rec',0)}  sender tx_count={f.get('n_tx',0)}",
        f"Sender age     : {f.get('age','N/A')}  city={f.get('city','')}",
        f"GPS vs home    : {f.get('gps_match','N/A')}  dist_km={f.get('gps_km',0)}",
        f"Phish SMS      : {f.get('phish_sms',0)}/3",
        f"Phish mail     : {f.get('phish_mail',0)}/3",
        f"SMS sample     : {str(f.get('sms_snip',''))[:200]}",
        f"Mail sample    : {str(f.get('mail_snip',''))[:200]}",
    ]
    return "\n".join(lines)


# ── Langfuse client (module-level) ────────────────────────────────────────────

lf = Langfuse(
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key = os.getenv("LANGFUSE_SECRET_KEY"),
    host       = os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
)
