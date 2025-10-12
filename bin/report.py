import glob
import os
import pandas as pd
from tqdm import tqdm


EXPERIMENTS = "enhanced"
DECODED = "test_decoded"


data = []


def read_stat_file(path):
    with open(path) as f:
        lines = f.readlines()

    line = lines[-1].split()
    return line


def parse_stat(path, name):
    d = read_stat_file(path)
    res = {}

    for x in d:
        if ":" not in x:
            continue

        k, v = x.split(":")

        if k in ("average", "All"):
            res[name] = float(v)
        else:
            res[name + "_" + k] = float(v)

    return res


def produce_result(d, ref):
    result = {}

    for k in d:
        result[k] = d[k]
        result["ref_" + k] = ref[k]
        result["d_" + k] = d[k] - ref[k]

    return result


def parse_params(p):
    p, q, a, d, s, _ = p.split("_")
    return {
        "profile": p,
        "qp": int(q[2:]),
        "alf": bool(int(a[3:])),
        "db": bool(int(d[2:])),
        "sao": bool(int(s[3:])),
    }


for experiment in tqdm(os.listdir(EXPERIMENTS)):
    print(f"processing experiment: {experiment}")
    exp_dir = os.path.join(EXPERIMENTS, experiment)

    for movie in tqdm(os.listdir(exp_dir)):
        mv_dir = os.path.join(exp_dir, movie)

        name = movie.split("__")[0]

        for params in glob.glob(f"{mv_dir}/*_420.yuv"):
            psnr_stats = params.replace(".yuv", "_psnr.info")
            ssim_stats = params.replace(".yuv", "_ssim.info")
            psnr = parse_stat(psnr_stats, "psnr")
            ssim = parse_stat(ssim_stats, "ssim")

            s = os.path.basename(params)
            dec = os.path.join(DECODED, f"{name}_{s}")
            psnr_stats = dec.replace(".yuv", "_psnr.info")
            ssim_stats = dec.replace(".yuv", "_ssim.info")
            ref_psnr = parse_stat(psnr_stats, "psnr")
            ref_ssim = parse_stat(ssim_stats, "ssim")

            data.append(
                dict(
                    **dict(
                        movie=name,
                        experiment=experiment,
                    ),
                    **parse_params(s),
                    **produce_result(psnr, ref_psnr),
                    **produce_result(ssim, ref_ssim),
                )
            )

df = pd.DataFrame(data)
df.to_csv("report.csv")
df.to_excel("report.xlsx")
