"""
Batch-create submission zips for all completed levels.

Usage: python create_all_submissions.py
       python create_all_submissions.py 1 2 3    (specific levels)
"""

import sys
import os
from create_submission import create_submission

levels = [str(l) for l in ([1, 2, 3, 4, 5] if len(sys.argv) == 1 else sys.argv[1:])]

print("Creating submission packages …\n")
created = []

for level in levels:
    output = f"output_level_{level}.txt"
    if not os.path.exists(output):
        print(f"  Level {level}: {output} not found — skipping")
        continue
    zip_name = create_submission(level)
    if zip_name:
        created.append((level, zip_name))

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
for level, zip_name in created:
    size_kb = os.path.getsize(zip_name) / 1024
    print(f"  Level {level}: {zip_name}  ({size_kb:.1f} KB)")

if not created:
    print("  No submissions created.")
    print("  Run 'python solve.py <level> eval' for each level first.")
