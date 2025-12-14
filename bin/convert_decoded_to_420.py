import os
import sys
from utils.read_metadata import read_movie_metadata

ROOT = "test_decoded"


TASKS = []


for movie in os.listdir(ROOT):
    if movie.endswith(".log"):
        continue

    if "420" in movie:
        continue

    name, height, width, frame_rate = read_movie_metadata(movie)
    name = movie.split(".")[0]

    source = f"{ROOT}/{movie}"
    target = f"{ROOT}/{name}_420.yuv"
    cmd = f"ffmpeg -y -pix_fmt yuv420p10le -s {width}x{height} -i {source} -pix_fmt yuv420p {target}"
    TASKS.append(cmd)


with open(f"tasks_convert_{ROOT}".replace("/", "_"), "w") as f:
    f.write("\n".join(TASKS))
