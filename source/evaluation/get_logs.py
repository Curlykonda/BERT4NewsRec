import os
import shutil
from pathlib import Path




if __name__ == "__main__":
    log_dir = Path("../../logs")
    if not log_dir.is_dir():
        log_dir.mkdir()

    exp_dir = Path("../../experiments")
    for exp in os.listdir(exp_dir):
        try:
            shutil.copytree(exp_dir.joinpath(exp + "/logs/"), log_dir.joinpath(exp))
        except:
            pass

