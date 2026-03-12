"""
Check Langfuse trace for a session ID.

Usage:
  python check_trace.py <session_id>
  python check_trace.py 1              → finds latest session file for level 1
  python check_trace.py 1 eval         → finds EVAL session file for level 1
  python check_trace.py 1 TRAIN_PREDICT → finds TRAIN_PREDICT session for level 1
"""

import os
import sys
import glob
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

client = Langfuse(
    public_key  = os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key  = os.getenv("LANGFUSE_SECRET_KEY"),
    host        = os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
)


def find_session_file(level: str, operation: str = None) -> str | None:
    """
    Find the most recent session file for a given level (and optional operation).
    File pattern: session_level_{level}_{operation}_{date}.txt
    """
    if operation:
        pattern = f"session_level_{level}_{operation.upper()}_*.txt"
    else:
        pattern = f"session_level_{level}_*.txt"
    matches = sorted(glob.glob(pattern), reverse=True)   # newest first (YYYYMMDD sort)
    return matches[0] if matches else None


def get_trace_info(session_id: str) -> dict | None:
    """Fetch all traces + observations for a session from Langfuse."""
    traces = []
    page   = 1
    while True:
        resp = client.api.trace.list(session_id=session_id, limit=100, page=page)
        if not resp.data:
            break
        traces.extend(resp.data)
        if len(resp.data) < 100:
            break
        page += 1

    if not traces:
        return None

    observations = []
    for trace in traces:
        detail = client.api.trace.get(trace.id)
        if detail and hasattr(detail, "observations"):
            observations.extend(detail.observations)

    if not observations:
        return None

    counts      : dict = defaultdict(int)
    costs       : dict = defaultdict(float)
    total_secs  : float = 0.0

    for obs in observations:
        if getattr(obs, "type", "") == "GENERATION":
            model = getattr(obs, "model", "unknown") or "unknown"
            counts[model] += 1
            cost = getattr(obs, "calculated_total_cost", None)
            if cost:
                costs[model] += float(cost)
            st = getattr(obs, "start_time", None)
            et = getattr(obs, "end_time",   None)
            if st and et:
                total_secs += (et - st).total_seconds()

    return {
        "traces":       len(traces),
        "observations": len(observations),
        "counts":       dict(counts),
        "costs":        dict(costs),
        "total_time_s": total_secs,
        "total_cost":   sum(costs.values()),
    }


def print_results(info: dict, session_id: str) -> None:
    print(f"\n{'='*60}")
    print(f"Session  : {session_id}")
    print(f"Traces   : {info['traces']}")
    print(f"Generations: {sum(info['counts'].values())}")
    print(f"{'─'*60}")
    if info["counts"]:
        print(f"{'Model':<45} {'Calls':>5}  {'Cost ($)':>10}")
        print(f"{'─'*60}")
        for model, cnt in sorted(info["counts"].items()):
            cost_str = f"{info['costs'].get(model, 0):.6f}"
            print(f"  {model:<43} {cnt:>5}  ${cost_str:>10}")
    print(f"{'─'*60}")
    print(f"  Total cost   : ${info['total_cost']:.6f}")
    print(f"  Total time   : {info['total_time_s']:.1f} s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_trace.py <session_id | level> [operation]")
        print("  python check_trace.py 1              # latest for level 1")
        print("  python check_trace.py 1 eval         # EVAL session for level 1")
        print("  python check_trace.py 1 TRAIN_PREDICT")
        sys.exit(1)

    arg = sys.argv[1]
    op  = sys.argv[2].upper() if len(sys.argv) > 2 else None

    if arg in [str(i) for i in range(1, 6)]:
        session_file = find_session_file(arg, op)
        if session_file:
            with open(session_file) as f:
                session_id = f.read().strip()
            print(f"Reading from : {session_file}")
        else:
            label = f"level {arg}" + (f" ({op})" if op else "")
            print(f"ERROR: no session file found for {label}")
            sys.exit(1)
    else:
        session_id = arg

    print(f"Querying session: {session_id}")
    info = get_trace_info(session_id)

    if info is None:
        print("No traces found for this session ID.")
        print("  • Check that LANGFUSE_PUBLIC_KEY / SECRET_KEY are correct in .env")
        print("  • The run may still be uploading — wait 30 s and retry")
    else:
        print_results(info, session_id)
