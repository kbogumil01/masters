import os
import sys
from utils.read_metadata import read_movie_metadata

if len(sys.argv) == 2:
    ROOT = sys.argv[1]
else:
    ROOT = "enhanced"


TASKS = []


for movie in os.listdir(ROOT):
    name, height, width, frame_rate = read_movie_metadata(movie)

    for params in os.listdir(os.path.join(ROOT, movie)):
        if "yuv" in params:
            continue

        source = f"{ROOT}/{movie}/{params}.yuv"
        target = f"{ROOT}/{movie}/{params}_420.yuv"
        cmd = f"ffmpeg -y -pix_fmt yuv444p -s {width}x{height} -i {source} -pix_fmt yuv420p {target}"
        TASKS.append(cmd)


with open(f"tasks_convert_{ROOT}".replace("/", "_"), "w") as f:
    f.write("\n".join(TASKS))
