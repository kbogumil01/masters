#!/usr/bin/env python3
# bin/create_decode_tasks.py
# Użycie:
#   python3 bin/create_decode_tasks.py TASKS_DEC [encoded_dir] [decoded_root]

import os
import sys

if len(sys.argv) == 4:
    encoded_dir = sys.argv[2]
    decoded_root = sys.argv[3]
else:
    encoded_dir = "videos/encoded"
    decoded_root = "videos/decoded"

out_list = sys.argv[1] if len(sys.argv) >= 2 else "TASKS_DEC"

vvcs = sorted(
    os.path.join(encoded_dir, f)
    for f in os.listdir(encoded_dir)
    if f.endswith(".vvc")
)

tasks = [f"bin/decode_data.sh {path} {decoded_root}" for path in vvcs]

with open(out_list, "w") as f:
    f.write("\n".join(tasks))

print(f"✔ Zapisano {out_list} ({len(tasks)} zadań)")
