"""
Stand-alone script to spawn local model training on a particular DAS node.
"""
import os
import pathlib
import subprocess
import sys
from time import sleep

from utils.args import get_args

if __name__ == "__main__":
    args = get_args()

    # Spawn process training on this node
    clients = args.dasclients.split(",")
    processes = []
    script_path = pathlib.Path(__file__).parent.resolve()
    os.chdir(script_path)
    for cur_client in clients:
        print("Spawning training subprocess for client %d...")
        p = subprocess.Popen(["python3", os.path.join(script_path, "train_local_model.py")] + sys.argv[1:] + ["--cindex", cur_client])
        processes.append(p)
        sleep(5)

    for p in processes:
        p.wait()
        if p.returncode != 0:
            raise RuntimeError("Training subprocess exited with non-zero code %d" % p.returncode)
