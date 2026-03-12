"""
Batch-create submission zips for all completed datasets.

Usage: python create_all_submissions.py
       python create_all_submissions.py truman deus brave    (specific datasets)
"""

import sys
import os
from create_submission import create_submission

ALL_DATASETS = ["truman", "deus", "brave", "1984", "blade"]
datasets = ALL_DATASETS if len(sys.argv) == 1 else [d.lower() for d in sys.argv[1:]]

print("Creating submission packages …\n")
created = []

for ds in datasets:
    output = f"output_{ds}.txt"
    if not os.path.exists(output):
        print(f"  {ds}: {output} not found — skipping")
        continue
    zip_name = create_submission(ds)
    if zip_name:
        created.append((ds, zip_name))

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
for ds, zip_name in created:
    size_kb = os.path.getsize(zip_name) / 1024
    print(f"  {ds}: {zip_name}  ({size_kb:.1f} KB)")

if not created:
    print("  No submissions created.")
    print("  Run 'python solve.py <dataset> eval' for each dataset first.")
