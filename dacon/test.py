import os
from PIL import Image
import pandas as pd
import numpy as np
import argparse
from time import strftime, time, localtime
import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as nnf
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.segformer import Segformer
from utils.dataloader import CustomDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modelB = Segformer(
    dims = (32, 64, 160, 256),      # dimensions of each stage
    heads = (1, 2, 5, 8),           # heads of each stage
    ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    num_layers = 2,                 # num layers of each stage
    decoder_dim = 256,              # decoder dimension
    num_classes = 13                 # number of segmentation classes
)
modelB.load_state_dict(torch.load('utils/mit_b1.pth'), strict=False)
ModelA = torch.load('utils/mit_b1.pth')
for param_tensor in modelB.state_dict():
    print(param_tensor, "\t", modelB.state_dict()[param_tensor].size())

print("---------")

print(ModelA)