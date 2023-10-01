#!/bin/bash
#SBATCH -J dacon_lu2_ex
#SBATCH --gres=gpu:1
#SBATCH --output=./out/Finetuning_lu1.0_th0.9_targetAg_lr000006.out
#SBATCH --time 0-45:00:00
mkdir -p ./out
eval "$(conda shell.bash hook)"
conda activate python3.9
python -u main_v2_lu_copy_ex.py --epoch 50 --resize 512 --lr 0.00006 --batch_size 4 --datadir ./dataset --outdir ./out --warmup 0