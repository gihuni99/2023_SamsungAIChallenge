#!/bin/bash
#SBATCH -J test_all_image
#SBATCH --gres=gpu:1
#SBATCH --output=./out/test_all_image.out
#SBATCH --time 0-3:00:00
mkdir -p ./out
eval "$(conda shell.bash hook)"
conda activate python3.9
python -u test_all_image.py