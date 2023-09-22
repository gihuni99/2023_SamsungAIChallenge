import os
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import argparse
from time import strftime, time, localtime
import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler
from torchvision import transforms
import torch.nn.functional as nnf
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR, ExponentialLR
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.segformer import SegFormer
from utils.dataloader import CustomDataset, Target
from utils.augseg import *
from utils.loss_helper import CriterionOhem
# 변경사항 : unsupervised loss 복원, ignore index, 사이즈 512


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
parser.add_argument("--ema_decay", type=int, default=0.996)
parser.add_argument("--debug_mode", type=bool, default=False)
args = parser.parse_args()


if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

tm = localtime(time())
starttime = strftime('%Y-%m-%d-%H:%M:%S',tm)

# start a new wandb run to track this script
if args.debug_mode == False:
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
        A.SomeOf([ # OneOf
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
        ], n=3, p=1)
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
student_model = SegFormer(
    in_channels=3,
    widths=[64, 128, 256, 512],
    depths=[3, 4, 6, 3],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    decoder_channels=256,
    scale_factors=[8, 4, 2, 1],
    num_classes=13,
).to(device)

# teacher_model 초기화
teacher_model = SegFormer(
    in_channels=3,
    widths=[64, 128, 256, 512],
    depths=[3, 4, 6, 3],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    decoder_channels=256,
    scale_factors=[8, 4, 2, 1],
    num_classes=13,
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
ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,eta_min=0.001)

#learning rate scheduler정의(CosineAnnealingWarmRestarts)
#scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=0, T_mult=2, eta_min=0.001)
#scheduler=CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=20, cycle_mult=1.0, 
#                                        max_lr=0.00006, min_lr=1e-12, warmup_steps=10, gamma=0.7)
scheduler = ExponentialLR(optimizer, gamma=0.98)
# scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
#                                             warmup_start_value=0.01,
#                                             warmup_end_value=0.1,
#                                             warmup_duration=3)
sup_loss_fn=CriterionOhem(aux_weight=0,thresh=0.7,min_kept=100000,ignore_index=12)

bestmIoU=0 #best_chekpoint 저장용
# training loop,

# for wandb
trf = A.Compose([A.Normalize()])

test_img = cv2.imread(os.path.join(args.datadir, 'train_target_image/TRAIN_TARGET_0000.png'))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = cv2.resize(test_img, (args.resize,args.resize))
test_img = trf(image=test_img)['image']
test_img = np.array(test_img)
test_img = torch.tensor(test_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

class_labels = {0: "Road", 1: "Sidewalk", 2: "Construction", 3: "Fence", 4: "Pole", 5: "Traffic Light",
                6: "Traffic Sign", 7: "Nature", 8: "Sky", 9: "Person", 10: "Rider", 11: "Car", 12: "BG"}

class_set = wandb.Classes(
    [
        {"name": "Road", "id": 0},
        {"name": "Sidewalk", "id": 1},
        {"name": "Construction", "id": 2},
        {"name": "Fence", "id": 3},
        {"name": "Pole", "id": 4},
        {"name": "Traffic Light", "id": 5},
        {"name": "Traffic Sign", "id": 6},
        {"name": "Nature", "id": 7},
        {"name": "Sky", "id": 8},
        {"name": "Person", "id": 9},
        {"name": "Rider", "id": 10},
        {"name": "Car", "id": 11},
        {"name": "BG", "id": 12},
    ]
)


for epoch in range(args.epochs):  # 에폭
    student_model.train()
    teacher_model.eval()
    ema_decay_origin = args.ema_decay
    warmup_epoch = args.warmup
    epoch_loss = 0
    l_x_loss = 0
    l_u_loss = 0
    data_length = min(len(dataloader), len(target_loader))
    if epoch < warmup_epoch:
        for images, masks in tqdm(dataloader):
            optimizer.zero_grad()
            images = images.float().to(device)
            masks = masks.long().to(device)
            p_y = student_model(images)
            p_y = nnf.interpolate(p_y, size=(args.resize, args.resize), mode='bicubic', align_corners=False)
            p_y = torch.softmax(p_y, dim=1)
            l_x = sup_loss_fn(p_y, masks)
            # l_x = F.cross_entropy(p_y, masks.squeeze(1), ignore_index=12, reduction="none")
            l_u = torch.tensor(0.0).cuda()
            
            loss = l_x + l_u

            # update student model
            # optimizer.zero_grad()
            loss.mean().backward()
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
            optimizer.zero_grad()
            #print(source_image.shape,source_mask.shape,target_weak,target_strong)
            # generate pseudo label
            with torch.no_grad():
                teacher_model.eval()
                pred_t = teacher_model(target_weak.detach()) #Ar(Aa)가 적용되지 않은 데이터가 들어가야 됨(Ag만 적용된)
                # print(f'target_weak:{target_weak}, pred_t:{pred_t}')
                #print(pred_t)
                pred_t = nnf.interpolate(pred_t, size=(args.resize, args.resize), mode='bicubic', align_corners=False)
                pred_t = torch.softmax(pred_t, dim=1)
                # p_t = torch.argmax(p_t, dim=1)
                p_t_logit, p_t = torch.max(pred_t, dim=1)
                #print('1','p_t:', p_t[0,0,0])
                # obtain confidence
                entropy = -torch.sum(pred_t * torch.log(pred_t + 1e-10), dim=1)
                entropy /= np.log(13)
                confidence = 1.0 - entropy
                confidence = confidence * p_t_logit
                confidence = confidence.mean(dim=[1,2])  # 1*C
                confidence = confidence.cpu().numpy().tolist()
                del pred_t
                # confidence = logits_u_aug.ge(p_threshold).float().mean(dim=[1,2]).cpu().numpy().tolist()
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
            del source_image, target_strong, target_weak
            pred_all = nnf.interpolate(pred_all, size=(args.resize, args.resize), mode='bicubic', align_corners=False)
            #pred_all = torch.softmax(pred_all, dim=1) #gihun

            pred_l = pred_all[:num_labeled]
            pred_u_strong = pred_all[num_labeled:]

            #debugging
            # if(epoch > 6):
            #     print('target_strong:', target_strong[1], 'pred_l:',pred_l[1], 'p_t:', p_t[1],
            #           'len_data', data_length, 'source_size', source_image.shape, 'strong_size', target_strong.shape,
            #           'i_iter', i_iter, 'ena', ema_decay, 'u_strong', pred_u_strong[0], 'pred_all', pred_all.shape)
                
            del pred_all
            # get supervised loss (l_x)
            # print("fffffffffff",pred_l.shape,source_mask.shape,source_mask.squeeze(1).shape)
            # [b 13 w h](logit), [b 1 w h]-> [b w h](label)
            
            l_x = sup_loss_fn(pred_l, source_mask)
            # print(f'mask:{source_mask.size()}, squeeze:{source_mask.squeeze(1).size()}')
            # l_x = F.cross_entropy(pred_l, source_mask, ignore_index=12, reduction="none")
            
            # get unsupervised loss (l_u)
            #print('2','epoch:',epoch, 'i_iter:', i_iter, 'source',source_mask[0,0,0],\
            #       'pred_l:', pred_l[0,0,0,0], 'p_t:', p_t[0,0,0])
            l_u, pseudo_high_ratio = compute_unsupervised_loss_by_threshold(pred_u_strong, p_t.detach(), p_t_logit.detach(), criterion=ce_loss,
                                                                            thresh=0.95, ignore_index=12)
            #l_u = ce_loss(pred_u_strong, p_t.detach())
            # l_u = nnf.cross_entropy(pred_u_strong, p_t.detach(), ignore_index=255, reduction="none") #gihun
            
            del pred_u_strong, pred_l, p_t, p_t_logit

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
            # optimizer.zero_grad()
            loss.mean().backward()
            # loss.mean().backward()
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
    teacher_model.eval()
    s_intersection_epoch = 0
    s_union_epoch = 0
    t_intersection_epoch = 0
    t_union_epoch = 0
    
    with torch.no_grad():
        test_out = student_model(test_img)
        test_out = nnf.interpolate(test_out, size=(args.resize, args.resize), mode='bicubic', align_corners = False)
        test_out = torch.softmax(test_out, dim=1).cpu()
        test_out = torch.argmax(test_out, dim=1).numpy() # b w h
        test_out = test_out.astype(np.uint8)
        test_out = test_out[0,:,:] # w h

    for image, mask in val_dataloader:
        image = image.float().to(device)
        mask = mask.long().to(device)
        with torch.no_grad():
            s_pred = student_model(image)
            t_pred = teacher_model(image)
        s_pred = nnf.interpolate(s_pred, size=(args.resize, args.resize), mode='bicubic', align_corners = False)
        t_pred = nnf.interpolate(t_pred, size=(args.resize, args.resize), mode='bicubic', align_corners = False)
        s_pred = torch.softmax(s_pred, dim=1).cpu()
        t_pred = torch.softmax(t_pred, dim=1).cpu()
        s_pred = torch.argmax(s_pred, dim=1).numpy()
        t_pred = torch.argmax(t_pred, dim=1).numpy()

        target_origin = mask.squeeze(1).cpu().numpy()
        
        s_intersection, s_union, target = intersectionAndUnion(s_pred, target_origin, 13, ignore_index=12)
        t_intersection, t_union, target = intersectionAndUnion(t_pred, target_origin, 13, ignore_index=12)
        s_intersection_epoch += s_intersection
        t_intersection_epoch += t_intersection
        s_union_epoch += s_union
        t_union_epoch += t_union
    s_iou_class = s_intersection_epoch/(s_union_epoch + 1e-10)
    t_iou_class = t_intersection_epoch/(t_union_epoch + 1e-10)
    s_mIoU = np.mean(s_iou_class)
    t_mIoU = np.mean(t_iou_class)
    ind, ct = np.unique(test_out, return_counts=True)
    if args.debug_mode == False:
        s_pred = s_pred.astype(np.uint8)
        s_pred = s_pred[0,:,:] # w h
        t_pred = t_pred.astype(np.uint8)
        t_pred = t_pred[0,:,:] # w h
        wandb.log({"loss": epoch_loss/data_length,
                "lr": optimizer.param_groups[0]["lr"],
                "l_x": l_x_loss/data_length,
                "l_u": l_u_loss/data_length,
                "student_mIoU": s_mIoU,
                "teacher_mIoU": t_mIoU})
        masked_image = wandb.Image(
        image[0],
        masks={
            "test": {"mask_data": test_out, "class_labels": class_labels},
            "s_pred": {"mask_data": s_pred, "class_labels": class_labels},
            "t_pred": {"mask_data": t_pred, "class_labels": class_labels},
        },
            classes=class_set,
        )

        ttt = wandb.Image(
        test_img,
        masks={
            "test": {"mask_data": test_out, "class_labels": class_labels},
        },
            classes=class_set,
        )

        table2 = wandb.Table(columns=["image"])
        table2.add_data(ttt)
        wandb.log({"field": table2})



        

    print(f'Epoch: {epoch+1}, loss: {epoch_loss/data_length}, \
            lr: {optimizer.param_groups[0]["lr"]}, student_mIoU: {s_mIoU}, \
            teacher_mIou: {t_mIoU},\
            l_x: {l_x_loss/data_length}, l_u: {l_u_loss/data_length}')

    if s_mIoU > bestmIoU:
        bestmIoU=s_mIoU
        torch.save(student_model.state_dict(), os.path.join(args.outdir, starttime + '_best.pt'))
        print(f'Model has been saved in {os.path.join(args.outdir, starttime)}_best.pt')
    else:
        torch.save(student_model.state_dict(), os.path.join(args.outdir, starttime + '_last.pt'))
        print(f'Model has been saved in {os.path.join(args.outdir, starttime)}_last.pt')

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