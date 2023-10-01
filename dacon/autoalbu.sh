#!/bin/bash
#SBATCH -J daycon2
#SBATCH --gres=gpu:1
#SBATCH --output=./out/autoalbu.out
#SBATCH --time 0-23:00:00
mkdir -p ./out
eval "$(conda shell.bash hook)"
conda activate python3.9
autoalbument-search --config-dir ./utils/autoalbument