#!/bin/bash
#SBATCH -J daycon2
#SBATCH --gres=gpu:1
#SBATCH --output=./out/dacon2.out
#SBATCH --time 0-23:00:00
mkdir -p ./out
eval "$(conda shell.bash hook)"
conda activate python3.9
python -u main_segf_augseg_gihun.py --epoch 9 --resize 256 --lr 0.00006 --batch_size 4 --datadir ./dataset --outdir ./out --warmup 0
