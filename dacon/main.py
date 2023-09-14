import os
import cv2
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

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.uNet import UNet
from utils.dataloader import CustomDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="Dacon")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--resize", type=int, default=224)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--datadir", type=str, default="./dataset")
parser.add_argument("--outdir", type=str, default="./out")
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

tm = localtime(time())
starttime = strftime('%Y-%m-%d-%H:%M:%S',tm)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="dacon-2023-09",
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": "Domain Adaptive Semantic Segmentation",
    "dataset": "dacon",
    "epochs": args.epochs,
    }
)



# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


transform = A.Compose(
    [   
        A.Resize(args.resize, args.resize),
        A.Normalize(),
        ToTensorV2()
    ]
)

dataset = CustomDataset(csv_file=  os.path.join(args.datadir, 'train_source.csv'), transform=transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)



# model 초기화
model = UNet().to(device)

# loss function과 optimizer 정의
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# training loop
for epoch in range(args.epochs):  # 에폭
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader):
        images = images.float().to(device)
        masks = masks.long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        print(outputs.size(), masks.size())
        loss = criterion(outputs, masks.squeeze(1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    wandb.log({"train_loss": epoch_loss/len(dataloader)})

    print(f'Epoch {epoch+1}, train_Loss: {epoch_loss/len(dataloader)}')

# Save Models
print(f'Model has been saved in {os.path.join(args.outdir, starttime)}.pt')
torch.save(model.state_dict(), os.path.join(args.outdir, starttime + '.pt'))



# Evaluation

test_dataset = CustomDataset(csv_file= os.path.join(args.datadir, 'test.csv'), transform=transform, infer=True, datadir=args.datadir)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)
        outputs = model(images)
        outputs = torch.softmax(outputs, dim=1).cpu()
        outputs = torch.argmax(outputs, dim=1).numpy()
        # batch에 존재하는 각 이미지에 대해서 반복
        for pred in outputs:
            pred = pred.astype(np.uint8)
            pred = Image.fromarray(pred) # 이미지로 변환
            pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
            pred = np.array(pred) # 다시 수치로 변환
            # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
            for class_id in range(12):
                class_mask = (pred == class_id).astype(np.uint8)
                if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                    mask_rle = rle_encode(class_mask)
                    result.append(mask_rle)
                else: # 마스크가 존재하지 않는 경우 -1
                    result.append(-1)





submit = pd.read_csv(os.path.join(args.datadir, 'sample_submission.csv'))
submit['mask_rle'] = result
submit

submit.to_csv(os.path.join(args.outdir, starttime + '.csv'), index=False)