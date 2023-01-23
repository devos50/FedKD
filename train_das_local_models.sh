#!/bin/bash

module load cuda11.7/toolkit/11.7
source /home/spandey/venv3/bin/activate
python3 train_das_local_models.py "$@"