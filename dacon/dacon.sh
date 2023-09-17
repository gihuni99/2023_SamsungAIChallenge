#!/bin/bash
#SBATCH -J daycon
#SBATCH --gres=gpu:1
#SBATCH --output=./out/dacon.out
#SBATCH --time 0-23:00:00
mkdir -p ./out
eval "$(conda shell.bash hook)"
conda activate python3.9
python -u main_segf_augseg.py --epoch 40 --lr 1e-3 --batch_size 16 --datadir ./dataset --outdir ./out
