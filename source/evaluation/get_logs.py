import argparse
import os
import shutil
from pathlib import Path




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=int, default=1)

    args = parser.parse_args()

    if args.local:
        log_dir = Path("../../logs")
        exp_dir = Path("../../experiments")
    else:
        log_dir = Path("./logs")
        exp_dir = Path("./experiments")

    if not log_dir.is_dir():
        log_dir.mkdir()


    for exp in os.listdir(exp_dir):
        try:
            shutil.copytree(exp_dir.joinpath(exp + "/logs/"), log_dir.joinpath(exp))
        except:
            pass

