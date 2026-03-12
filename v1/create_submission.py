"""
Create submission zip for one challenge level.

Usage: python create_submission.py <level>
       python create_submission.py 1

Packages: output_level_{level}.txt + source files + session ID file.
Prints the Session ID you must paste in the upload form.
"""

import sys
import os
import glob
import zipfile
from datetime import datetime


def create_submission(level: str) -> str | None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name  = f"submission_level_{level}_{timestamp}.zip"

    print(f"\n{'='*60}")
    print(f"Creating submission for Level {level} …")
    print(f"{'='*60}")

    # --- Output file ---
    output_file = f"output_level_{level}.txt"
    if not os.path.exists(output_file):
        print(f"ERROR: {output_file} not found.")
        print(f"  Run first: python solve.py {level} eval")
        return None

    # --- Find EVAL session file ---
    session_file = None
    session_id   = None
    date_str = datetime.now().strftime("%Y%m%d")
    candidate = f"session_level_{level}_EVAL_{date_str}.txt"
    if os.path.exists(candidate):
        session_file = candidate
    else:
        # Fall back: any EVAL session file for this level
        matches = sorted(glob.glob(f"session_level_{level}_EVAL_*.txt"), reverse=True)
        if matches:
            session_file = matches[0]
            print(f"  Note: using session file from earlier date: {session_file}")

    if session_file:
        with open(session_file) as f:
            session_id = f.read().strip()
    else:
        print(f"  WARNING: no EVAL session file found for level {level}.")
        print(f"  Make sure to run: python solve.py {level} eval")

    # --- Files to zip ---
    source_files = [
        "main.py",
        "solve.py",
        "data_agent.py",
        "requirements.txt",
        "README.md",
        output_file,
    ]
    if session_file:
        source_files.append(session_file)

    # --- Create zip ---
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in source_files:
            if os.path.exists(fname):
                zf.write(fname)
                print(f"  ✓ {fname}")
            else:
                print(f"  ⚠ missing: {fname}")

    zip_kb = os.path.getsize(zip_name) / 1024
    print(f"\n✓ Created : {zip_name}  ({zip_kb:.1f} KB)")

    # --- Summary ---
    print(f"\n{'─'*60}")
    print(f"[Session ID] → paste this in the upload form:")
    print(f"  {session_id or '(not found)'}")
    print(f"{'─'*60}")

    if session_id:
        print("\n[Verifying Langfuse trace …]")
        os.system(f"python check_trace.py {session_id}")

    print(f"\n[Next steps]")
    print(f"  1. Upload: {zip_name}  (evaluation dataset submission)")
    print(f"  2. Paste session ID above in the form")
    print(f"  3. Submit — ONE submission only, irreversible!")

    return zip_name


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_submission.py <level>")
        sys.exit(1)
    create_submission(sys.argv[1])
