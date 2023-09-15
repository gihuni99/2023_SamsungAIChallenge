#!/bin/bash
#SBATCH -J test
#SBATCH --gres=gpu:1
#SBATCH --output=./out/test.out
#SBATCH --time 0-23:00:00
eval "$(conda shell.bash hook)"
conda activate python3.9
python -u test.py
