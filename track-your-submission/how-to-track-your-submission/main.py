"""
Langfuse trace checker - query and analyze traces by session ID.

Usage: python main.py <session_id>

Langfuse automatically tracks your agent's token usage, costs, and latency. 
You just need to set up the environment variables (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST), 
use the @observe() decorator and CallbackHandler() in your code, and generate a unique session ID for each run. 
Langfuse handles the rest. For the full setup and code examples, refer to 
"Resource Management & Toolkit for the Challenge" in the Learn & Train section.
"""

import os
import sys
from dotenv import load_dotenv
from langfuse import Langfuse
from datetime import datetime
from collections import defaultdict

load_dotenv()

if not all([os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")]):
    raise ValueError("Missing Langfuse credentials in .env")

client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)


def get_trace_info(session_id):
    traces = []
    page = 1

    while True:
        response = client.api.trace.list(session_id=session_id, limit=100, page=page)
        if not response.data:
            break
        traces.extend(response.data)
        if len(response.data) < 100:
            break
        page += 1

    if not traces:
        return None

    observations = []
    for trace in traces:
        detail = client.api.trace.get(trace.id)
        if detail and hasattr(detail, 'observations'):
            observations.extend(detail.observations)

    if not observations:
        return None

    sorted_obs = sorted(
        observations,
        key=lambda o: o.start_time if hasattr(o, 'start_time') and o.start_time else datetime.min
    )

    counts = defaultdict(int)
    costs = defaultdict(float)
    total_time = 0

    for obs in observations:
        if hasattr(obs, 'type') and obs.type == 'GENERATION':
            model = getattr(obs, 'model', 'unknown') or 'unknown'
            counts[model] += 1

            if hasattr(obs, 'calculated_total_cost') and obs.calculated_total_cost:
                costs[model] += obs.calculated_total_cost

            if hasattr(obs, 'start_time') and hasattr(obs, 'end_time'):
                if obs.start_time and obs.end_time:
                    total_time += (obs.end_time - obs.start_time).total_seconds()

    first_input = ""
    if sorted_obs and hasattr(sorted_obs[0], 'input'):
        inp = sorted_obs[0].input
        if inp:
            first_input = str(inp)[:100]

    last_output = ""
    if sorted_obs and hasattr(sorted_obs[-1], 'output'):
        out = sorted_obs[-1].output
        if out:
            last_output = str(out)[:100]

    return {
        'counts': dict(counts),
        'costs': dict(costs),
        'time': total_time,
        'input': first_input,
        'output': last_output
    }


def print_results(info):
    if not info:
        print("\nNo traces found for this session\n")
        return

    print("\nTrace Count by Model:")
    for model, count in info['counts'].items():
        print(f"  {model}: {count}")

    print("\nCost by Model:")
    total = 0
    for model, cost in info['costs'].items():
        print(f"  {model}: ${cost:.6f}")
        total += cost
    if total > 0:
        print(f"  Total: ${total:.6f}")

    print(f"\nTotal Time: {info['time']:.2f}s")

    if info['input']:
        print(f"\nInitial Input:\n  {info['input']}")

    if info['output']:
        print(f"\nFinal Output:\n  {info['output']}")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]
    print(f"\nQuerying session: {session_id}")

    try:
        info = get_trace_info(session_id)
        print_results(info)
    except Exception as e:
        print(f"\nError: {e}\n")
        sys.exit(1)
