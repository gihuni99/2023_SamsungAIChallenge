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
from utils.model_helper import *
from utils.network.modeling import _segm_resnet # DeeplabV3

from transformers import SegformerForSemanticSegmentation
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


## To do
# main_v2_lu_copy.py, dacon2_lu_copy.sh
# 1. 최적 환경(coef) 적어두기 (다시 찾아야 된다)
# 2. BG밑에 왜곡 인식 문제 -> Affine, Optical같은 augmentation 활용
# 3. 밤 11시 30분에 제출 3번(후보 3개 그 전에 상의해서 정하기)
# 4. 기존 DeepLab, 새로운 DeepLab 비교
# 5. 



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
parser.add_argument("--ema_decay", type=int, default=0.99) #0.996
parser.add_argument("--debug_mode", type=bool, default=False)
parser.add_argument("--ignore", type=int, default=255)
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

# height = 1080
# width = 1920
# back_mask = np.zeros((height, width), dtype=np.uint8)
# pts = np.array([[200,0], [93,223], [50, 370], [96, 484], [90, 500], [110,750], [223, 813], [410, 980], [400,935], [582,1019], [740, 1056],
#                 [width//2, height],[width-630, 1042],[width-350, 943], [width-223, 813], [width-110, 750],
#                 [width-53, 520], [width-9, 400],[width-52, 223],[width-170,0],
#                 ], np.int32)
# cv2.fillPoly(back_mask,[pts], 255)
# road_mask = np.zeros((height, width), dtype=np.uint8)
# road = np.array([[225, 813], [500, 950], [738,1011], [width//2, 1030], [width-738, 1011], [width-500, 950],[width-223,813]], np.int32)
# cv2.fillPoly(road_mask,[road], 255)
# back_mask = cv2.resize(back_mask, (args.resize, args.resize))
# back_mask = torch.tensor(back_mask, dtype=torch.bool)
# back_mask = torch.stack([back_mask for i in range(0, args.batch_size)], dim=0).to(device)
# road_mask = cv2.resize(road_mask, (args.resize, args.resize))
# road_mask = torch.tensor(road_mask, dtype=torch.bool)
# road_mask = torch.stack([road_mask for i in range(0, args.batch_size)], dim=0).to(device)



# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

#A_g(weak_geometrical augmentation)
transform_A_g = A.Compose(
    [   
        #A.RandomSizedCrop(min_max_height=[512,1024], height=args.resize, width=args.resize,  #고려해보아야함
        #                    w2h_ratio=1.0, interpolation=1, always_apply=False,p=0.5),
        A.HorizontalFlip(always_apply=False,p=0.5),
        A.VerticalFlip(always_apply=False,p=0.5),
        #A.RandomScale(always_apply=False, scale_limit=(0.8, 1.2),p=0.5),
        # A.Cutout(always_apply=True, p=0.5, num_holes=5, max_h_size=28, max_w_size=28),
        A.Resize(args.resize, args.resize) #수정 필요(resize 따로하고 이후에 Augmentation하는 것이 더 좋아보임)
    ]
)
    
t_csv_file= os.path.join(args.datadir, 'train_target.csv')
t_data = pd.read_csv(t_csv_file)
t_img_path=[]
for i in range(0,2922):
    t_img_path.append(os.path.join(args.datadir,t_data.iloc[i,1]))


transform_A_g_strong = A.Compose(
    [  
        # A.RandomSizedCrop(min_max_height=[512,1024], height=args.resize, width=args.resize,  #고려해보아야함
                        #    w2h_ratio=1.0, interpolation=1, always_apply=False,p=0.5),
        #A.Cutout(always_apply=True, p=0.5, num_holes=5, max_h_size=28, max_w_size=28),
        # A.OneOf([
        #         A.RandomCrop(200, 200, always_apply=True),
        #         A.RandomCrop(300, 300, always_apply=True),
        #         A.RandomCrop(400, 400, always_apply=True),
        #         A.RandomCrop(500, 500, always_apply=True),
        #         A.RandomCrop(600, 600, always_apply=True),
        #         A.RandomCrop(700, 700, always_apply=True),
        #         A.RandomCrop(800, 800, always_apply=True)
        #      ],p=0.4),
        A.RandomResizedCrop(512, 512, scale=(0.2, 1.0), ratio=(0.75, 1.25), 
                                interpolation=1, always_apply=False, p=0.5),
        A.Rotate(limit=[-180, 180], interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0,
                 mask_value=12, rotate_method='largest_box', crop_border=False, always_apply=True),
        A.SomeOf([ # OneOf
            A.HorizontalFlip(always_apply=True),
            A.VerticalFlip(always_apply=True),
            A.ElasticTransform (alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_CUBIC, border_mode=4, value=None,
                              mask_value=None, always_apply=True, approximate=False, same_dxdy=False),
            A.OneOf([
                # A.FDA(reference_images = t_img_path, beta_limit=0.1, 
                #      read_fn=lambda x:cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB),always_apply=True),
                A.Perspective(scale=(0.05, 0.8), 
                               keep_size=True, 
                               pad_mode=3, #가장자리 채우는 모드(reflect)
                               pad_val=None, 
                               mask_pad_val=None,
                            fit_output=False, 
                            interpolation=1, 
                            always_apply=True),
                A.OpticalDistortion(distort_limit=0.2, #왜곡 정도
                                    shift_limit=0.2, #이동 정도
                                    interpolation=1, #interpolation
                                    border_mode=4, #왜곡된 영역의 가장자리(reflect)
                                    value=None, #채우기 사용
                                    mask_value=None, #mask fill
                                    always_apply=True)
             ],p=1),
            A.RandomScale(always_apply=True, scale_limit=(0.8, 1.2)),
            A.Rotate(limit=90, interpolation=1, 
                     border_mode=4, value=None, 
                     mask_value=None, 
                     rotate_method='largest_box', 
                     crop_border=False, 
                     always_apply=True),
            A.Cutout(always_apply=True, num_holes=3, max_h_size=50, max_w_size=50),
        ], n=4, p=1),
        A.Resize(args.resize, args.resize),
    ]
)
# transform_A_g = A.Compose(
#     [   
#         A.Resize(args.resize, args.resize), #수정 필요(resize 따로하고 이후에 Augmentation하는 것이 더 좋아보임)
#         A.HorizontalFlip(always_apply=False, p=0.5),



#         # A.RandomScale(always_apply=True, scale_limit=(0.5, 1.0)),
#         # A.Cutout(always_apply=True, p=0.5, num_holes=5, max_h_size=28, max_w_size=28),
#         #A.RandomCrop(height=1024, width=1024,always_apply=True),
#     ]
# )

#A_r(random intensity-based augmentation)
transform_A_r = A.Compose( #augmentation중에서  줄이고 우리한테 적합한 것 추가
    [   
        A.SomeOf([ # OneOf
            A.ColorJitter(brightness=0,contrast=0,saturation=0, hue=0, always_apply=True),
            A.RandomContrast (limit=0.2, always_apply=True), #Autocontrast
            A.Equalize(mode='cv', always_apply=True), #Histogram Equalization
            A.GaussianBlur(always_apply=True), #Gaussian blur
            #A.ColorJitter(brightness=0,contrast=(0.5,0.95),saturation=0, hue=0, always_apply=True), #Contrast
            A.Sharpen(alpha=(0.5, 0.95), always_apply=True), #Sharpness
            A.ColorJitter(brightness=0,contrast=0,saturation=(1.05,1.95), hue=0, always_apply=True), #Color
            A.ColorJitter(brightness=(0.05,0.95),contrast=0,saturation=0, hue=0, always_apply=True), #Brightness
            A.ColorJitter(brightness=0,contrast=0,saturation=0, hue=(0,0.5), always_apply=True), #Hue
            A.Posterize(always_apply=True),
            A.Solarize (always_apply=True),
            A.GaussNoise(var_limit=(10, 50), mean=0, per_channel=True, always_apply=True),
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


dataset = CustomDataset(csv_file=  os.path.join(args.datadir, 'train_source.csv'), mode='train',
                        transform=transform_A_g_strong, datadir=args.datadir)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
target_data = Target(csv_file= os.path.join(args.datadir, 'train_target.csv'), transform=transform_A_g, transfrom_r=transform_A_r, datadir=args.datadir)
target_loader = DataLoader(target_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True) 

val_dataset = CustomDataset(csv_file=  os.path.join(args.datadir, 'val_source.csv'), mode='val', transform=test_transform, datadir=args.datadir)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_dataset = CustomDataset(csv_file= os.path.join(args.datadir, 'test.csv'), mode='test', transform=test_transform, datadir=args.datadir)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)



#state_dict = torch.load('./utils/pretrained/best_deeplabv3plus_resnet101_cityscapes_os16.pth')

# student_model 초기화
#student_model=_segm_resnet(name='deeplabv3plus', backbone_name='resnet101', num_classes=13, output_stride=16, pretrained_backbone=False).to(device)
#missing_keys, unexpected_keys = student_model.load_state_dict(state_dict, strict=False)
#print(
        #     f"[Info]  pretrain model",
        #     "\nmissing_keys: ",
        #     missing_keys,
        #     "\nunexpected_keys: ",
        #     unexpected_keys,
        # )
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

pretrained_model_name = "nvidia/mit-b4"#add_segformer
student_model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
).to(device)

teacher_model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
).to(device)

#student_model=ModelBuilder().to(device)

# student_model = SegFormer(
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

# # teacher_model 초기화
#teacher_model=_segm_resnet(name='deeplabv3plus', backbone_name='resnet101', num_classes=13, output_stride=16, pretrained_backbone=False).to(device)
#missing_keys, unexpected_keys = teacher_model.load_state_dict(state_dict, strict=False)
#teacher_model=ModelBuilder().to(device)


# teacher_model = SegFormer(
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



# loss function과 optimizer 정의
ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
#optimizer = torch.optim.Adam(student_model.parameters(), lr=0.00006,betas=(0.9, 0.999), weight_decay=0.01) #weight_decay설정(실험적으로 찾는 값)
optimizer = torch.optim.SGD(student_model.parameters(), momentum=0.9,lr=args.lr,weight_decay=0.0005) #0.0005
scheduler = CosineAnnealingWarmupRestarts(optimizer, 
                                          first_cycle_steps=10, 
                                          cycle_mult=1.0, 
                                          max_lr=0.001, 
                                          min_lr=0.000000001,
                                          warmup_steps=3, 
                                          gamma=0.7)
#scheduler = ExponentialLR(optimizer, gamma=0.98)
sup_loss_fn=CriterionOhem(aux_weight=0,thresh=0.7,min_kept=100000,ignore_index=args.ignore)
bestmIoU=0
# training loop,

# for wandb
trf = A.Compose([A.Normalize()])

target_img_o = cv2.imread(os.path.join(args.datadir, 'train_target_image/TRAIN_TARGET_0743.png'))
test_img_o = cv2.imread(os.path.join(args.datadir, 'test_image/TEST_0284.png'))
val_img_o = cv2.imread(os.path.join(args.datadir, 'val_source_image/VALID_SOURCE_211.png'))
val_mask_o = cv2.imread(os.path.join(args.datadir, 'val_source_gt/VALID_SOURCE_211.png'), cv2.IMREAD_GRAYSCALE)
target_img_o = cv2.resize(target_img_o, (args.resize, args.resize))
test_img_o = cv2.resize(test_img_o, (args.resize, args.resize))
val_img_o = cv2.resize(val_img_o, (args.resize, args.resize))
val_mask_o = cv2.resize(val_mask_o, (args.resize, args.resize))
val_mask_o[val_mask_o == 255] = 12 #배경을 픽셀값 12로 간주

target_img_o = cv2.cvtColor(target_img_o, cv2.COLOR_BGR2RGB)
test_img_o = cv2.cvtColor(test_img_o, cv2.COLOR_BGR2RGB)
val_img_o = cv2.cvtColor(val_img_o, cv2.COLOR_BGR2RGB)
target_img = trf(image=target_img_o)['image']
val_img = trf(image=val_img_o)['image']
test_img = trf(image=test_img_o)['image']
target_img = np.array(target_img)
val_img = np.array(val_img)
test_img = np.array(test_img)
target_img = torch.tensor(target_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
test_img = torch.tensor(test_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
val_img = torch.tensor(val_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

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
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            l_x_loss += l_x.item()
            l_u_loss += l_u.item()
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
            # generate pseudo label
            with torch.no_grad():
                teacher_model.eval()
                pred_t = teacher_model(target_weak.detach()) #(loss, logits, hidden_states, attentions)
                pred_t=pred_t['logits']#add_segformer
                pred_t = nnf.interpolate(pred_t, size=(args.resize, args.resize), mode='bicubic', align_corners=False)
                pred_t = torch.softmax(pred_t, dim=1)
                p_t_logit, p_t = torch.max(pred_t, dim=1)
                #print(p_t_logit[1][:][:20])
                #p_t[~back_mask] = 12
                #p_t_logit[~back_mask] = 0.8
                #p_t[~road_mask] = 0
                #p_t_logit[~road_mask] = 0.6
                #p_t_logit[~back_mask] = 0.951 #실험 필요
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
            
            # 3. forward concate labeled + unlabeld into student networks
            num_labeled = len(source_image)
            pred_all = student_model(torch.cat((source_image, target_strong), dim=0))
            del source_image, target_strong, target_weak
            pred_all=pred_all['logits']#add_segformer
            pred_all = nnf.interpolate(pred_all, size=(args.resize, args.resize), mode='bicubic', align_corners=False)

            pred_l = pred_all[:num_labeled]
            pred_u_strong = pred_all[num_labeled:]
            # my_thresh=0.5
            # if epoch>9 and epoch<=14:
            #     my_thresh=0.6
            # elif epoch>14 and epoch<=19:
            #     my_thresh=0.7
            # elif epoch>19 and epoch<=24:
            #     my_thresh=0.8
            # elif epoch>24 and epoch<=29:
            #     my_thresh=0.9
            # elif epoch>29:
            #     my_thresh=0.95
            
            del pred_all
            l_x = sup_loss_fn(pred_l, source_mask)
            l_u, pseudo_high_ratio = compute_unsupervised_loss_by_threshold(pred_u_strong, p_t.detach(), p_t_logit.detach(), criterion=ce_loss,
                                                                            thresh=0.5, ignore_index=args.ignore)
            
            # l_u = sup_loss_fn(pred_u_strong, background(p_t.detach(), mask, mask_inv))
            # l_u = nnf.cross_entropy(pred_u_strong, p_t.detach(), ignore_index=10, reduction="none")
            # l_u = l_u.mean() #l_u값은 [b h w]형태 (gihun)
            del pred_u_strong, pred_l, p_t, p_t_logit

            with torch.no_grad():
                ema_decay = min(1- 1/ (i_iter - data_length * warmup_epoch+ 1), ema_decay_origin)

            loss = l_x + 0.5*l_u
            # update student model
            loss.backward()
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
            epoch_loss += loss.item()
            l_x_loss += l_x.item()
            l_u_loss += l_u.item()
    scheduler.step()

    # validation
    student_model.eval()
    teacher_model.eval()
    s_intersection_epoch = 0
    s_union_epoch = 0
    t_intersection_epoch = 0
    t_union_epoch = 0

    for image, mask in val_dataloader:
        image = image.float().to(device)
        mask = mask.long().to(device)
        with torch.no_grad():
            s_pred = student_model(image)
            t_pred = teacher_model(image)
            s_pred=s_pred['logits'] #add_segformer
            t_pred=t_pred['logits'] #add_segformer
        s_pred = nnf.interpolate(s_pred, size=(args.resize, args.resize), mode='bicubic', align_corners = False)
        t_pred = nnf.interpolate(t_pred, size=(args.resize, args.resize), mode='bicubic', align_corners = False)
        s_pred = torch.softmax(s_pred, dim=1).cpu()
        t_pred = torch.softmax(t_pred, dim=1).cpu()
        s_pred = torch.argmax(s_pred, dim=1).numpy()
        t_pred = torch.argmax(t_pred, dim=1).numpy()

        target_origin = mask.squeeze(1).cpu().numpy()
        
        s_intersection, s_union, target = intersectionAndUnion(s_pred, target_origin, 13, ignore_index=args.ignore)
        t_intersection, t_union, target = intersectionAndUnion(t_pred, target_origin, 13, ignore_index=args.ignore)
        s_intersection_epoch += s_intersection
        t_intersection_epoch += t_intersection
        s_union_epoch += s_union
        t_union_epoch += t_union
    s_iou_class = s_intersection_epoch/(s_union_epoch + 1e-10)
    t_iou_class = t_intersection_epoch/(t_union_epoch + 1e-10)
    s_mIoU = np.mean(s_iou_class)
    t_mIoU = np.mean(t_iou_class)
    if args.debug_mode == False: # for logging wandb panels
        with torch.no_grad():
            s_pred = student_model(val_img)
            t_pred = teacher_model(val_img)
            s_pred=s_pred['logits'] #add_segformer
            t_pred=t_pred['logits'] #add_segformer
            test_out = student_model(test_img)
            target_out = student_model(target_img)
            test_t_out = teacher_model(test_img)
            test_out=test_out['logits'] #add_segformer
            target_out=target_out['logits'] #add_segformer
            test_t_out=test_t_out['logits'] #add_segformer
            s_pred = nnf.interpolate(s_pred, size=(args.resize, args.resize), mode='bicubic', align_corners = False)
            t_pred = nnf.interpolate(t_pred, size=(args.resize, args.resize), mode='bicubic', align_corners = False)
            test_out = nnf.interpolate(test_out, size=(args.resize, args.resize), mode='bicubic', align_corners = False)
            target_out = nnf.interpolate(target_out, size=(args.resize, args.resize), mode='bicubic', align_corners = False)
            test_t_out = nnf.interpolate(test_t_out, size=(args.resize, args.resize), mode='bicubic', align_corners = False)
            s_pred = torch.softmax(s_pred, dim=1).cpu()
            t_pred = torch.softmax(t_pred, dim=1).cpu()
            test_out = torch.softmax(test_out, dim=1).cpu()
            target_out = torch.softmax(target_out, dim=1).cpu()
            test_t_out = torch.softmax(test_t_out, dim=1).cpu()
            s_pred = torch.argmax(s_pred, dim=1).numpy()
            t_pred = torch.argmax(t_pred, dim=1).numpy()
            test_out = torch.argmax(test_out, dim=1).numpy()
            target_out = torch.argmax(target_out, dim=1).numpy()
            test_t_out = torch.argmax(test_t_out, dim=1).numpy()
            s_pred = s_pred.astype(np.uint8)
            t_pred = t_pred.astype(np.uint8)
            test_out = test_out.astype(np.uint8)
            target_out = target_out.astype(np.uint8)
            test_t_out = test_t_out.astype(np.uint8)
            s_pred = s_pred.squeeze()
            t_pred = t_pred.squeeze()
            test_out = test_out.squeeze()
            target_out = target_out.squeeze()
            test_t_out = test_t_out.squeeze()

            wandb.log({"loss": epoch_loss/data_length,
                    "lr": optimizer.param_groups[0]["lr"],
                    "l_x": l_x_loss/data_length,
                    "l_u": l_u_loss/data_length,
                    "student_mIoU": s_mIoU,
                    "teacher_mIoU": t_mIoU,
                    "s_pred": wandb.Image(val_img_o, masks={
                        "s_pred": {"mask_data": s_pred, "class_labels": class_labels}},
                        classes=class_set),
                    "gt:": wandb.Image(val_img_o, masks={
                        "gt": {"mask_data": val_mask_o, "class_labels": class_labels}},
                        classes=class_set),
                    "t_pred:": wandb.Image(val_img_o, masks={
                        "t_pred": {"mask_data": t_pred, "class_labels": class_labels}},
                        classes=class_set),
                    "test_s": wandb.Image(test_img_o, masks={
                        "test_s": {"mask_data": test_out, "class_labels": class_labels}},
                        classes=class_set),
                    "target": wandb.Image(target_img_o, masks={
                        "target": {"mask_data": target_out, "class_labels": class_labels}},
                        classes=class_set),
                    "test_t": wandb.Image(test_img_o, masks={
                        "test_t": {"mask_data": test_t_out, "class_labels": class_labels}},
                        classes=class_set),
                    })

    print(f'Epoch: {epoch+1}, loss: {epoch_loss/data_length}, \
            lr: {optimizer.param_groups[0]["lr"]}, student_mIoU: {s_mIoU}, \
            teacher_mIou: {t_mIoU},\
            l_x: {l_x_loss/data_length}, l_u: {l_u_loss/data_length}')

    if s_mIoU > bestmIoU:
        bestmIoU=s_mIoU
        torch.save(student_model.state_dict(), os.path.join(args.outdir, starttime + '_s_best.pt'))
        torch.save(teacher_model.state_dict(), os.path.join(args.outdir, starttime + '_t_best.pt'))
        print(f'Model has been saved in {os.path.join(args.outdir, starttime)}_best.pt')
    else:
        torch.save(student_model.state_dict(), os.path.join(args.outdir, starttime + '_s_last.pt'))
        torch.save(teacher_model.state_dict(), os.path.join(args.outdir, starttime + '_t_last.pt'))
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