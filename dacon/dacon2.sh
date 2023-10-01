#!/bin/bash
#SBATCH -J daycon2
#SBATCH --gres=gpu:1
#SBATCH --output=./out/dacon2.out
#SBATCH --time 0-23:00:00
mkdir -p ./out
eval "$(conda shell.bash hook)"
conda activate python3.9
python -u main_v2.py --epoch 50 --resize 512 --lr 0.00006 --batch_size 8 --datadir ./dataset --outdir ./out --warmup 0
