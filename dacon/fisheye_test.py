import cv2
import albumentations as A
import numpy as np
    
test_img_o = cv2.imread('./dataset/test_image/TEST_0284.png', cv2.IMREAD_GRAYSCALE)
# test_img_o = cv2.resize(test_img_o, (512,512))ㅊ
target_img_o = cv2.imread('./dataset/train_target_image/TRAIN_TARGET_2153.png', cv2.IMREAD_GRAYSCALE)
# target_img_o = cv2.resize(target_img_o, (512,512))

# mask = source + res
height = 1080
width = 1920
back_mask = np.zeros((height, width), dtype=np.uint8)
pts = np.array([[200,0], [93,223], [50, 370], [96, 484], [90, 500], [110,750], [223, 813], [410, 980], [400,935], [582,1019], [740, 1056],
                [width//2, height],[width-630, 1042],[width-350, 943], [width-223, 813], [width-110, 750],
                [width-53, 520], [width-9, 400],[width-52, 223],[width-170,0],
                ], np.int32)
cv2.fillPoly(back_mask,[pts], 255)
# pts = np.array([[270,0], [145,223], [90, 500], [110,750],
#                 [500,950], [738,1011], [width//2, 1030], [width-738, 1011], [width-500, 950],[width-90,750],[width-70, 500], 
#                 [width-105, 223],[width-200,0],
#                 ], np.int32)
road_mask = np.zeros((height, width), dtype=np.uint8)
road = np.array([[225, 813], [500, 950], [738,1011], [width//2, 1030], [width-738, 1011], [width-500, 950],[width-223,813]], np.int32)
cv2.fillPoly(road_mask,[road], 50)
# back_mask = cv2.resize(back_mask, (512, 512))
ret = target_img_o + back_mask*0.1 + road_mask

# cv2.ellipse(target, (height//2, width//2), (height//2, width//2), 0, 0, 360, (255,255,0), thickness=-1)

cv2.imwrite("./out/test.png", ret)