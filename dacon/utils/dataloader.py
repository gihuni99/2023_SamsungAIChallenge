import cv2
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

img_tr=transforms.ToPILImage()

def image_unnorm(image):
    # 정규화된 이미지 데이터
    normalized_image =image
    normalized_image=normalized_image.cpu()
    normalized_image=np.array(normalized_image,dtype=np.uint8)
    normalized_image=np.transpose(normalized_image,(1,2,0))#[c h w]->[h w c](transform_A_r의 input형식)
    # 평균과 표준 편차
    #mean = (0.485, 0.456, 0.406)
    #std = (0.229, 0.224, 0.225)
    #restored_image = normalized_image * np.array(std) + np.array(mean)
    #restored_image = np.clip(restored_image, 0, 1)
    #restored_image = (restored_image * 255).astype(np.uint8)
    return normalized_image #restored_image



class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, mode='train', datadir="./dataset"):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.mode = mode
        self.datadir = datadir
        self.tr = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
        assert transform!=None, "parameter (transform) must be specified."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.datadir, self.data.iloc[idx, 1])
        image = cv2.imread(img_path)
        # cv2.imwrite("./test_img/origin_source.png", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode=='test' and self.transform:
            # cv2.imwrite("./test_img/origin_test.png", image)
            image = self.transform(image=image)['image']
            return image
        
        mask_path = os.path.join(self.datadir, self.data.iloc[idx, 2])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12 #배경을 픽셀값 12로 간주

        if self.mode=='val' and self.transform:
            # cv2.imwrite("./test_img/origin_val.png", image)
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        elif self.mode=='train' and self.transform:
            augmented = self.transform(image=image, mask=mask)
            # cv2.imwrite("./test_img/weak_source.png", augmented['image'])
            augmented = self.tr(image=augmented['image'], mask=augmented['mask'])
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
    

class Target(Dataset):
    def __init__(self, csv_file, transform=None, transfrom_r=None, datadir="./dataset"):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.transform_r = transfrom_r
        self.datadir = datadir
        self.tr = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
        assert self.transform!=None and self.transform_r!=None, "A_g and A_r must be specified for the Target dataloader"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.datadir, self.data.iloc[idx, 1])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform and self.transform_r:
            # cv2.imwrite("./test_img/origin_target.png", image)
            augmented = self.transform(image=image)
            image = augmented['image']
            # cv2.imwrite("./test_img/weak_target.png", image)
            strong = self.transform_r(image=image)['image']
            # cv2.imwrite("./test_img/strong_target.png", strong)
            image = self.tr(image=image)['image']
            strong = self.tr(image=strong)['image']

        return image, strong