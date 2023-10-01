#!/bin/bash
#SBATCH -J dacon_lu
#SBATCH --gres=gpu:1
#SBATCH --output=./out/emadecay09_lucoef05_mask06_th05_pretrainsegformer.out
#SBATCH --time 0-23:00:00
mkdir -p ./out
eval "$(conda shell.bash hook)"
conda activate python3.9
python -u main_v2_lu_segformer.py --epoch 100 --resize 512 --lr 0.01 --batch_size 8 --datadir ./dataset --outdir ./out --warmup 0
