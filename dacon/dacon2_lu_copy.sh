#!/bin/bash
#SBATCH -J dacon_lu2
#SBATCH --gres=gpu:1
#SBATCH --output=./out/NewAugSetting_noFDA_b4.out
#SBATCH --time 0-45:00:00
mkdir -p ./out
eval "$(conda shell.bash hook)"
conda activate python3.9
python -u main_v2_lu_copy.py --epoch 50 --resize 512 --lr 0.001 --batch_size 4 --datadir ./dataset --outdir ./out --warmup 0