import os
import sys


if len(sys.argv) == 4:
    data_dir = sys.argv[2]
    target_dir = sys.argv[3]
else:
    data_dir = "data"
    target_dir = "encoded"


mkvs = [x for x in os.listdir(data_dir) if x.endswith("yuv")]
profiles = ["AI"]
QP = 28, 32, 37, 42, 47


tasks = []

for m in mkvs:
    for p in profiles:
        for q in QP:
            for p1 in (0, 1):
                for p2 in (0, 1):
                    for p3 in (0, 1):
                        tasks.append(
                            f"bin/encode_data.sh {p} {q} {p1} {p2} {p3} {data_dir}/{m} {target_dir}"
                        )

with open(sys.argv[1], "w") as f:
    f.write("\n".join(tasks))
