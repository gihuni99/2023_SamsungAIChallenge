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
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.segformer import Segformer
from utils.dataloader import CustomDataset, Target
from utils.augseg import *

# 1. Data증강 부분 분리/함수 추가/제거 + Fisheye << 이거에 집착하지말고 가장 나중에..
# 2. Pytorch 호환되는지 확인 후 Segformer 부분 HuggingFace 이용해서 불러오고. 적용 (전이학습)
# 3. Learning Rate Scheduler 구현

# 4 (같이 해도 좋음). 실험. 적정 에포크/segformer 모델 사이즈. 하이퍼파라미터 튜닝



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(17)

parser = argparse.ArgumentParser(description="Dacon")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--resize", type=int, default=224)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--datadir", type=str, default="./dataset")
parser.add_argument("--outdir", type=str, default="./out")
parser.add_argument("--warmup", type=int, default=0, help="0 or 1")
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

#A_g(weak_geometrical augmentation):
#A_r(random intensity-based augmentation): 최대 k개 적용한다.(k도 최대값만을 설정하고 random한 값)
#A_a(adaptive label-injecting augmentation):adaptive label-injecting CutMix augmentation이다. 논문 Figure 4에 나와 있음
# A_r을 단일로 사용하는 것보다.(A_r(A_a))로 사용하는 것이 더 좋은 성능이 나왔다고 주장함.
#cut_mix augmentation은 논문 github에서 코드 따와야 할 것 같다.

transform_A_g = A.Compose(
    [   
        A.RandomScale(always_apply=True, scale_limit=[0.5,1.0])
        A.HorizontalFlip(always_apply=False, p=0.5),
        A.RandomCrop(height=448, width=448,always_apply=True)
    ]
)

transform_A_r = A.Compose(
    [   
        A.SomeOf([
            A.ColorJitter(brightness=0,contrast=0,saturation=0, hue=0, always_apply=True) #Identity
            A.ColorJitter(brightness=0,contrast=(2,2),saturation=0, hue=0, always_apply=True) #Autocontrast
            A.Equalize(mode='cv', by_channels=True, always_apply=True) #Histogram Equalization
            A.GaussianBlur(always_apply=True) #Gaussian blur
            A.ColorJitter(brightness=0,contrast=(0.05,0.95),saturation=0, hue=0, always_apply=True) #Contrast
            A.harpen(alpha=(0.05, 0.95), always_apply=True) #Sharpness
            A.ColorJitter(brightness=0,contrast=0,saturation=(1.05,1.95), hue=0, always_apply=True) #Color
            A.ColorJitter(brightness=(0.05,0.95),contrast=0,saturation=0, hue=0, always_apply=True) #Brightness
            A.ColorJitter(brightness=0,contrast=0,saturation=0, hue=(0,0.5), always_apply=True) #Hue
            A.posterize(bits=4)
            A.solarize (threshold=128)
        ], n=3),
        A.Normalize(),
        ToTensorV2()
    ]
)


dataset = CustomDataset(csv_file=  os.path.join(args.datadir, 'train_source.csv'), transform=transform_A_g)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
val_dataset = CustomDataset(csv_file=  os.path.join(args.datadir, 'val_source.csv'), transform=transform_A_g)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
target_data = Target(csv_file= os.path.join(args.datadir, 'train_target.csv'), transform=transform_A_g)
target_loader = DataLoader(target_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True) 


# student_model 초기화
student_model = Segformer(
    dims = (64, 128, 320, 512),      # dimensions of each stage
    heads = (1, 2, 5, 8),           # heads of each stage
    ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    num_layers = 2,                 # num layers of each stage
    decoder_dim = 512,              # decoder dimension
    num_classes = 13                 # number of segmentation classes
).to(device)
# teacher_model 초기화
teacher_model = Segformer(
    dims = (64, 128, 320, 512),      # dimensions of each stage
    heads = (1, 2, 5, 8),           # heads of each stage
    ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    num_layers = 2,                 # num layers of each stage
    decoder_dim = 512,              # decoder dimension
    num_classes = 13                 # number of segmentation classes
).to(device)


for p in teacher_model.parameters():
    p.requires_grad = False

# initialize teacher model -- not neccesary if using warmup
with torch.no_grad():
    for t_params, s_params in zip(teacher_model.parameters(), student_model.parameters()):
        t_params.data = s_params.data


# Print model's state_dict
print("Model's state_dict:")
for param_tensor in student_model.state_dict():
    print(param_tensor, "\t", student_model.state_dict()[param_tensor].size())

## To do list
## tensor dim! squeeze 



# loss function과 optimizer 정의
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)

# training loop
for epoch in range(args.epochs):  # 에폭
    student_model.train()
    teacher_model.eval()
    ema_decay_origin = 0.999
    warmup_epoch = args.warmup
    epoch_loss = 0
    l_x_loss = 0
    l_u_loss = 0
    data_length = min(len(dataloader), len(target_loader))
    if epoch < warmup_epoch: # 0 또는 1로 하는거같음.
        for images, masks in tqdm(dataloader):
            images = images.float().to(device)
            masks = masks.long().to(device)
            p_y = student_model(images) # tensor[b 13 w/4 h/4] << score 값. 각 픽셀에서 각 클래스마다 
            p_y = nnf.interpolate(p_y, size=(args.resize, args.resize), mode='bicubic', align_corners=True)
            # tensor [b 13 w h]
            # mask > gray scale image  tensor [b 1 w h]
            l_x = ce_loss(p_y, masks.squeeze(1))
            # tensor [b w h]
    
            l_u = torch.tensor(0.0).cuda()
            
            loss = l_x + l_u

            # update student model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            l_x_loss += l_x.item()
            l_u_loss += l_u.item()
            with torch.no_grad():
                ema_decay = 0.0 # 첫번째 epoch: student => teacher copy

#여기에서는 targetdata를 augmentation한 것과 안한 것을 따로 불러와야 함
#따라서 teacher와 student에 사용할 target data를 각각의 dataloader로 따로 불러와야 한다.
    else: # now starts augseg 
        for i, data in enumerate(zip(tqdm(dataloader), target_loader)):
            i_iter = epoch  * data_length + i # 지금까지 총 iteration
            # get the inputs; data is a list of [inputs, labels]
            source, target_image = data
            source_image, source_mask = source
            target_image = target_image.float().to(device)
            source_image = source_image.float().to(device)
            source_mask = source_mask.long().to(device)
            # generate pseudo label
            with torch.no_grad():
                teacher_model.eval()
                pred_t = teacher_model(target_image.detach()) #Ar(Aa)가 적용되지 않은 데이터가 들어가야 됨(Ag만 적용된)
                pred_t = nnf.interpolate(pred_t, size=(args.resize, args.resize), mode='bicubic', align_corners=True)
                pred_t = torch.softmax(pred_t, dim=1)
                # p_t = torch.argmax(p_t, dim=1)
                p_t_logit, p_t = torch.max(pred_t, dim=1)

                # obtain confidence
                entropy = -torch.sum(pred_t * torch.log(pred_t + 1e-10), dim=1)
                entropy /= np.log(13)
                confidence = 1.0 - entropy
                confidence = confidence * p_t_logit
                confidence = confidence.mean(dim=[1,2])  # 1*C
                confidence = confidence.cpu().numpy().tolist()
                # confidence = logits_u_aug.ge(p_threshold).float().mean(dim=[1,2]).cpu().numpy().tolist()
                del pred_t
            student_model.train()
            #todo: apply addptive cutmix
            if np.random.uniform(0,1) < 0.5:
                target_image, p_t, p_t_logit = cut_mix_label_adaptive(target_image, p_t, p_t_logit, source_image, source_mask, confidence)
            # 3. forward concate labeled + unlabeld into student networks
            num_labeled = len(source_image) # b
            #여기에는 targetdata가 augmentation(Ar(Aa))된 데이터가 들어가야 된다.
            target_image=transform_A_r(target_image) #apply random intensity-based augmentation
            pred_all = student_model(torch.cat((source_image, target_image), dim=0)) # ( 2*b, 13, w/4, h/4) 
            pred_all = nnf.interpolate(pred_all, size=(args.resize, args.resize), mode='bicubic', align_corners=True)
            # pred all [2b, 13, w, h]
            pred_l= pred_all[:num_labeled] # 0~b
            pred_u_strong = pred_all[num_labeled:] # b~2b

            # get supervised loss (l_x)
            l_x = ce_loss(pred_l, source_mask)
            
            # get unsupervised loss (l_u)
            l_u, pseudo_high_ratio = compute_unsupervised_loss_by_threshold(pred_u_strong, p_t.detach(), p_t_logit.detach(),
                                                                            thresh=0.95)

            with torch.no_grad():
                ema_decay = min(1- 1/ (i_iter - data_length * warmup_epoch+ 1), ema_decay_origin)
            
            loss = l_x + l_u

            # update student model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update teacher model with EMA
            with torch.no_grad():
                for param_train, param_eval in zip(student_model.parameters(), teacher_model.parameters()):
                    param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
                # update bn
                for buffer_train, buffer_eval in zip(student_model.buffers(), teacher_model.buffers()):
                    buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)
                    # buffer_eval.data = buffer_train.data
            epoch_loss += loss.item()
            l_x_loss += l_x.item()
            l_u_loss += l_u.item()


    # validation
    student_model.eval()
    intersection_epoch = 0
    union_epoch = 0
    for image, mask in val_dataloader:
        image = image.float().to(device)
        mask = mask.long().to(device)
        with torch.no_grad():
            pred = student_model(image)
        pred = nnf.interpolate(pred, size=(args.resize, args.resize), mode='bicubic', align_corners = True)
        pred = torch.softmax(pred, dim=1).cpu()
        pred = torch.argmax(pred, dim=1).numpy()
        target_origin = mask.cpu().numpy()
        
        intersection, union, target = intersectionAndUnion(pred, target_origin, 13, ignore_index=12)
        intersection_epoch += intersection
        union_epoch += union
    iou_class = intersection_epoch/(union_epoch + 1e-10)
    mIoU = np.mean(iou_class)

    wandb.log({"loss": epoch_loss/data_length,
               "l_x": l_x_loss/data_length,
               "l_u": l_u_loss/data_length,
               "mIoU": mIoU})

    print(f'Epoch {epoch+1}, mIoU: {mIoU}')

    torch.save(student_model.state_dict(), os.path.join(args.outdir, starttime + '.pt'))
    print(f'Model has been saved in {os.path.join(args.outdir, starttime)}.pt')

# Evaluation

test_dataset = CustomDataset(csv_file= os.path.join(args.datadir, 'test.csv'), transform=transform, infer=True, datadir=args.datadir)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

with torch.no_grad():
    student_model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)
        outputs = student_model(images)
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