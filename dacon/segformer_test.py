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
from torch.utils.data import RandomSampler
from torchvision import transforms
import torch.nn.functional as nnf
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR, ExponentialLR
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.segformer import SegFormer
from utils.dataloader import CustomDataset, Target
from utils.augseg import *
from einops import rearrange
from torchvision.utils import save_image
from utils.model_helper import *
from transformers import SegformerForSemanticSegmentation

tm = localtime(time())
starttime = strftime('%Y-%m-%d-%H:%M:%S',tm)

test_transform = A.Compose(
    [   
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2()
    ]
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(17)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


test_dataset = CustomDataset(csv_file= './dataset/test.csv', mode='test', transform=test_transform, datadir='./dataset')
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

id2label= {#add_segformer
    0: "Road",
    1: "Sidewalk",
    2: "Construction",
    3: "Fence",
    4: "Pole",
    5: "Traffic Light",
    6: "Traffic Sign",
    7: "Nature",
    8: "Sky",
    9: "Person",
    10: "Rider",
    11: "Car",
    12: "BG"
}

label2id= {
    "Road": 0,
    "Sidewalk": 1,
    "Construction": 2,
    "Fence": 3,
    "Pole": 4,
    "Traffic Light": 5,
    "Traffic Sign": 6,
    "Nature": 7,
    "Sky": 8,
    "Person": 9,
    "Rider": 10,
    "Car": 11,
    "BG": 12
}

pretrained_model_name = "nvidia/mit-b5"#add_segformer
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
).to(device)


#model checkpoint load
#model=student_model=ModelBuilder().to(device)
# model = student_model = SegFormer(
#     in_channels=3,
#     widths=[64, 128, 256, 512],
#     depths=[3, 4, 6, 3],
#     all_num_heads=[1, 2, 4, 8],
#     patch_sizes=[7, 3, 3, 3],
#     overlap_sizes=[4, 2, 2, 2],
#     reduction_ratios=[8, 4, 2, 1],
#     mlp_expansions=[4, 4, 4, 4],
#     decoder_channels=256,
#     scale_factors=[8, 4, 2, 1],
#     num_classes=13,
# ).to(device)
checkpoint = torch.load('./out/test_ckpt_save/Segformer_lr0.01_wd0.0005_lu0.2_11epoch.pt')
model.load_state_dict(checkpoint)

# Evaluation
with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)
        outputs = model(images)
        outputs=outputs['logits']#add_segformer
        outputs = nnf.interpolate(outputs, size=(960, 540), mode='bicubic', align_corners=True)
        outputs = torch.softmax(outputs, dim=1).cpu()
        outputs = torch.argmax(outputs, dim=1).numpy()
        # batch에 존재하는 각 이미지에 대해서 반복
        for pred in outputs:
            pred = pred.astype(np.uint8)
            #print(pred)
            pred = Image.fromarray(pred) # 이미지로 변환
            #pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
            pred = np.array(pred) # 다시 수치로 변환
            # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
            for class_id in range(12):
                class_mask = (pred == class_id).astype(np.uint8)
                if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                    mask_rle = rle_encode(class_mask)
                    result.append(mask_rle)
                else: # 마스크가 존재하지 않는 경우 -1
                    result.append(-1)


submit = pd.read_csv('./dataset/sample_submission.csv')
submit['mask_rle'] = result
submit

submit.to_csv(os.path.join('./out', starttime + '.csv'), index=False)