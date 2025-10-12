from utils.determine_orig import determine_orig
from utils.read_metadata import read_movie_metadata
import glob
import sys
import os

if len(sys.argv) == 2:
    ROOT = sys.argv[1]
else:
    ROOT = "enhanced"


CMD_TEMPLATE = """ffmpeg
-pix_fmt yuv420p
-s {res}
-i {reffile}
-pix_fmt yuv420p
-s {res}
-i {infile}
-filter_complex "{filter}"
-f null /dev/null
> {logfile}
2>&1""".replace(
    "\n", " "
)

FILTERS = ["ssim", "psnr"]


TASKS = []


def produce_tasks(f: str) -> None:
    global TASKS
    name, height, width, _ = read_movie_metadata(f)
    orig_fpath = determine_orig(name)

    for filter in FILTERS:
        logfile = f.replace(".yuv", f"_{filter}.info")
        TASKS.append(
            CMD_TEMPLATE.format(
                res=f"{width}x{height}",
                reffile=orig_fpath,
                infile=f,
                filter=filter,
                logfile=logfile,
            )
        )


if ROOT.startswith("enhanced"):
    for movie in glob.glob(f"{ROOT}/*/*_420.yuv"):
        produce_tasks(movie)
else:
    for movie in glob.glob(f"{ROOT}/*_420.yuv"):
        produce_tasks(movie)


with open(f"tasks_metrics_{ROOT.replace(os.sep, '_')}", "w") as f:
    f.write("\n".join(TASKS))
