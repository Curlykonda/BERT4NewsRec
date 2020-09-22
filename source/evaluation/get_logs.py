import argparse
import os
import shutil
from pathlib import Path




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=int, default=1)
    parser.add_argument("--incl_model", type=int, default=0)
    parser.add_argument("--specified_only", type=int, default=0, help="Get logs only for specified exps in 'logs_to_get.txt' or all")
    parser.add_argument("--target_dir", type=str, default="logs")

    args = parser.parse_args()

    if args.local:
        root = "../../"
    else:
        root = "./"

    log_dir = Path(root + args.target_dir)
    exp_dir = Path(root + "experiments")

    if not log_dir.is_dir():
        log_dir.mkdir()

    exps = []

    if args.specified_only:
        # load exp names from json file
        with open(root + "logs_to_get.txt", 'r') as fin:
            for line in fin.readlines():
                exps.append(line.strip())
    else:
        exps = os.listdir(exp_dir)

    copied = 0
    for exp in exps:
        try:
            shutil.copy(exp_dir.joinpath(exp + "/config.json"), log_dir.joinpath(exp))
            shutil.copytree(exp_dir.joinpath(exp + "/logs/"), log_dir.joinpath(exp))

            if args.incl_model:
                shutil.copytree(exp_dir.joinpath(exp + "/models/"), log_dir.joinpath(exp, "models"))
            copied += 1
        except Exception as e:
            print(e)

    print("\n copied {} new logs".format(copied))
