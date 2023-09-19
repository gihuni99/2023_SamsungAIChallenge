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
from utils.segformer import Segformer
from utils.dataloader import CustomDataset, Target
from utils.augseg import *
from einops import rearrange
from torchvision.utils import save_image

#https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts #lr scheduler

img_tr=transforms.ToPILImage()

def image_unnorm(image):
    # 정규화된 이미지 데이터
    normalized_image =image
    normalized_image=normalized_image.cpu()
    normalized_image=np.array(normalized_image,dtype=np.uint8)
    normalized_image=np.transpose(normalized_image,(1,2,0))#[c h w]->[h w c](transform_A_r의 input형식)
    # 평균과 표준 편차
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    #restored_image = normalized_image * np.array(std) + np.array(mean)
    #restored_image = np.clip(restored_image, 0, 1)
    #restored_image = (restored_image * 255).astype(np.uint8)
    return normalized_image#restored_image



# 1. Data증강 부분 분리/함수 추가/제거 + Fisheye << 이거에 집착하지말고 가장 나중에.. (구현 완료, but 목적에 맞게 수정 필요)
# 2. Pytorch 호환되는지 확인 후 Segformer 부분 HuggingFace 이용해서 불러오고. 적용 (전이학습) (loss 문제 해결 이후 구현 예정)
# 3. Learning Rate Scheduler 구현 (구현 완료, cosine이 적합한지는 실험적으로 확인)
# 4 (같이 해도 좋음). 실험. 적정 에포크/segformer 모델 사이즈. 하이퍼파라미터 튜닝

#Loss Problem 가능성
#1. 너무 높은 LR
#2. LR Scheduler
#3. Loss function 자체



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(17)

parser = argparse.ArgumentParser(description="Dacon")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--resize", type=int, default=512) #size를 조금 더 늘려볼 수 있도록 해야함(512에 BS=8이면 괜찮음)
parser.add_argument("--lr", type=float, default=0.01)
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

#A_g(weak_geometrical augmentation)
#A_r(random intensity-based augmentation): 최대 k개 적용한다.(논문에서 k=3)
#A_a(adaptive label-injecting augmentation):adaptive label-injecting CutMix augmentation이다. 논문 Figure 4에 나와 있음
# A_r을 단일로 사용하는 것보다.(A_r(A_a))로 사용하는 것이 더 좋은 성능이 나왔다고 주장함.

#A_g(weak_geometrical augmentation)
transform_A_g = A.Compose(
    [   
        A.Resize(args.resize, args.resize), #수정 필요(resize 따로하고 이후에 Augmentation하는 것이 더 좋아보임)
        A.HorizontalFlip(always_apply=False, p=0.5),
        # A.RandomScale(always_apply=True, scale_limit=(0.5, 1.0)),
        # A.Cutout(always_apply=True, p=0.5, num_holes=5, max_h_size=28, max_w_size=28),
        #A.RandomCrop(height=1024, width=1024,always_apply=True),
    ]
)
#A_r(random intensity-based augmentation)
transform_A_r = A.Compose( #augmentation중에서 줄일거는 줄이고 우리한테 적합한 것 추가
    [   
        A.OneOf([ # one of 가 나은듯.
            A.ColorJitter(brightness=0,contrast=0,saturation=0, hue=0, always_apply=True), #Identity
            A.ColorJitter(brightness=0,contrast=(2,2),saturation=0, hue=0, always_apply=True), #Autocontrast
            A.Equalize(mode='cv', always_apply=True), #Histogram Equalization
            A.GaussianBlur(always_apply=True), #Gaussian blur
            A.ColorJitter(brightness=0,contrast=(0.05,0.95),saturation=0, hue=0, always_apply=True), #Contrast
            A.Sharpen(alpha=(0.05, 0.95), always_apply=True), #Sharpness
            A.ColorJitter(brightness=0,contrast=0,saturation=(1.05,1.95), hue=0, always_apply=True), #Color
            A.ColorJitter(brightness=(0.05,0.95),contrast=0,saturation=0, hue=0, always_apply=True), #Brightness
            A.ColorJitter(brightness=0,contrast=0,saturation=0, hue=(0,0.5), always_apply=True), #Hue
            A.Posterize(always_apply=True),
            A.Solarize (always_apply=True),
        ], p=1)
    ]
)
test_transform = A.Compose(
    [   
        A.Resize(args.resize, args.resize),
        A.Normalize(),
        ToTensorV2()
    ]
)

dataset = CustomDataset(csv_file=  os.path.join(args.datadir, 'train_source.csv'), transform=transform_A_g, datadir=args.datadir)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
target_data = Target(csv_file= os.path.join(args.datadir, 'train_target.csv'), transform=transform_A_g, transfrom_r=transform_A_r, datadir=args.datadir)
target_loader = DataLoader(target_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True) 

val_dataset = CustomDataset(csv_file=  os.path.join(args.datadir, 'val_source.csv'), mode='val', transform=test_transform, datadir=args.datadir)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_dataset = CustomDataset(csv_file= os.path.join(args.datadir, 'test.csv'), mode='test', transform=test_transform, datadir=args.datadir)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# student_model 초기화
student_model = Segformer(
    dims = (32, 64, 160, 256),      # dimensions of each stage
    heads = (1, 2, 5, 8),           # heads of each stage
    ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    num_layers = 2,                 # num layers of each stage
    decoder_dim = 256,              # decoder dimension
    num_classes = 13                 # number of segmentation classes
).to(device)

# teacher_model 초기화
teacher_model = Segformer(
    dims = (32, 64, 160, 256),      # dimensions of each stage
    heads = (1, 2, 5, 8),           # heads of each stage
    ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    num_layers = 2,                 # num layers of each stage
    decoder_dim = 256,              # decoder dimension
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
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=12, reduction="none")
optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,eta_min=0.001)

#learning rate scheduler정의(CosineAnnealingWarmRestarts)
#scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=0, T_mult=2, eta_min=0.001)
#scheduler=CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=20, cycle_mult=1.0, 
#                                        max_lr=1e-6, min_lr=1e-9, warmup_steps=5, gamma=0.8)
# torch_lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.98)
# scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
#                                             warmup_start_value=0.01,
#                                             warmup_end_value=0.1,
#                                             warmup_duration=3)
# training loop,
for epoch in range(args.epochs):  # 에폭
    student_model.train()
    teacher_model.eval()
    ema_decay_origin = 0.999
    warmup_epoch = args.warmup
    epoch_loss = 0
    l_x_loss = 0
    l_u_loss = 0
    data_length = min(len(dataloader), len(target_loader))
    if epoch < warmup_epoch:
        for images, masks in tqdm(dataloader):
            images = images.float().to(device)
            masks = masks.long().to(device)
            p_y = student_model(images)
            p_y = nnf.interpolate(p_y, size=(args.resize, args.resize), mode='bicubic', align_corners=True)
            l_x = ce_loss(p_y, masks.squeeze(1))
            # l_x = F.cross_entropy(p_y, masks.squeeze(1), ignore_index=12, reduction="none")
            l_u = torch.tensor(0.0).cuda()
            
            loss = l_x + l_u

            # update student model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.mean()
            l_x_loss += l_x.mean()
            l_u_loss += l_u.mean()
            with torch.no_grad():
                ema_decay = 0.0 # 첫번째 epoch: student => teacher copy

    else: # now starts augseg 
        for i, data in enumerate(zip(tqdm(dataloader), target_loader)):
            i_iter = epoch  * data_length + i # 지금까지 총 iteration
            # get the inputs; data is a list of [inputs, labels]
            source, target = data
            source_image, source_mask = source
            target_weak, target_strong = target
            source_image = source_image.float().to(device)
            source_mask = source_mask.long().to(device)
            target_weak = target_weak.float().to(device)
            target_strong = target_strong.float().to(device)
            # generate pseudo label
            with torch.no_grad():
                teacher_model.eval()
                pred_t = teacher_model(target_weak.detach()) #Ar(Aa)가 적용되지 않은 데이터가 들어가야 됨(Ag만 적용된)
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
            #apply addptive cutmix(adaptive label-injecting CutMix augmentation)

            if np.random.uniform(0,1) < 0.5:
                target_strong, p_t, p_t_logit = cut_mix_label_adaptive(target_strong, p_t, p_t_logit, source_image, source_mask.squeeze(1), confidence)
                # save_image(target_strong, './test_img/cut_mix_target.png')
            #A_r(random intensity-based augmentation)구현 코드
            #target_image_ar은 target_image를 np.array로 바꾸어 Ar을 적용하기 위한 변수
            #target_image를 Tensor 변수로 남겨놓기 위함
            #image_un=image_unnorm(target_image[0].squeeze())
            #img=img_tr(image_un)
            #img.save('./test_img/Cutmix_target.png')
            # target_image_ar=np.array(target_image_ar,dtype=np.uint8).cpu()#numpy[b c h w]
            # for i in range(len(target_image)):#batch size만큼 반복
            #     target_array=[] #data[c h w]를 저장해 Ar을 적용하기 위한 변수
            #     target_array=target_image_ar[i].squeeze() #[b c h w] -> [c h w]
            #     target_array=np.transpose(target_array,(1,2,0))#[c h w]->[h w c](transform_A_r의 input형식)
            #     target_array=transform_A_r(image=target_array) # [h w c]->[c h w]
            #     target_array=target_array['image'] #[c h w]
            #     target_image[i]=target_array #target_image: [b c h w]
                #image_un=image_unnorm(target_array)
                #img=img_tr(image_un)
                #img.save('./test_img/Ar_Aa_target.png')

            
            # 3. forward concate labeled + unlabeld into student networks
            num_labeled = len(source_image) # b
            #여기에는 targetdata가 augmentation(Ar(Aa))된 데이터가 들어가야 된다.
            pred_all = student_model(torch.cat((source_image, target_strong), dim=0))
            pred_all = nnf.interpolate(pred_all, size=(args.resize, args.resize), mode='bicubic', align_corners=True)
            pred_l= pred_all[:num_labeled]
            pred_u_strong = pred_all[num_labeled:]

            # get supervised loss (l_x)
            l_x = ce_loss(pred_l, source_mask.squeeze(1))
            # print(f'mask:{source_mask.size()}, squeeze:{source_mask.squeeze(1).size()}')
            # l_x = F.cross_entropy(pred_l, source_mask, ignore_index=12, reduction="none")
            
            # get unsupervised loss (l_u)
            l_u, pseudo_high_ratio = compute_unsupervised_loss_by_threshold(pred_u_strong, p_t.detach(), p_t_logit.detach(), criterion=ce_loss,
                                                                            thresh=0.95, ignore_index=12)

            with torch.no_grad():
                # i_iter = epoch  * data_length + step
                # warmup_epoch = 0 or 1
                ema_decay = min(1- 1/ (i_iter - data_length * warmup_epoch+ 1), ema_decay_origin)
                # 0 ~ increasing
            
            loss = l_x + l_u
            # if loss.mean() > 50:
            #     print(f'anormaly detected! loss automatically set 0  at loss: {loss.mean()}, epoch: {epoch}, ema_decay: {ema_decay}')
            #     loss = torch.tensor(0.0).cuda()

            # update student model
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            # update teacher model with EMA
            with torch.no_grad():
                for param_train, param_eval in zip(student_model.parameters(), teacher_model.parameters()):
                    param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
                # update bn
                for buffer_train, buffer_eval in zip(student_model.buffers(), teacher_model.buffers()):
                    buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)
                    # buffer_eval.data = buffer_train.data
            epoch_loss += loss.mean()
            l_x_loss += l_x.mean()
            l_u_loss += l_u.mean()
    scheduler.step()

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
        target_origin = mask.squeeze(1).cpu().numpy()
        
        intersection, union, target = intersectionAndUnion(pred, target_origin, 13, ignore_index=12)
        intersection_epoch += intersection
        union_epoch += union
    iou_class = intersection_epoch/(union_epoch + 1e-10)
    mIoU = np.mean(iou_class)

    wandb.log({"loss": epoch_loss/data_length,
               "lr": optimizer.param_groups[0]["lr"],
               "l_x": l_x_loss/data_length,
               "l_u": l_u_loss/data_length,
               "mIoU": mIoU})

    print(f'Epoch: {epoch+1}, loss: {epoch_loss/data_length}, \
            lr: {optimizer.param_groups[0]["lr"]}, mIoU: {mIoU}, \
            l_x: {l_x_loss/data_length}, l_u: {l_u_loss/data_length}')

    torch.save(student_model.state_dict(), os.path.join(args.outdir, starttime + '.pt'))
    print(f'Model has been saved in {os.path.join(args.outdir, starttime)}.pt')

# Evaluation

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