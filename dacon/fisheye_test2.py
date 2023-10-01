import albumentations as A
import cv2
import numpy as np


def apply(image, mask):

    mask_cropped = np.zeros(mask.shape, dtype=np.uint8)
    img_cropped = np.zeros(image.shape, dtype=np.uint8)
    mask_cropped[mask == 255] = mask[mask==255]
    img_cropped[mask==255] = image[mask==255]
    mask_cropped_left = cv2.rotate(mask_cropped, cv2.ROTATE_90_CLOCKWISE)
    mask_cropped_right = cv2.rotate(mask_cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_cropped_left = cv2.rotate(img_cropped, cv2.ROTATE_90_CLOCKWISE)
    img_cropped_right = cv2.rotate(img_cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    mask_cropped_left = cv2.resize(mask_cropped_left, (image.shape[1], image.shape[0]))
    mask_cropped_right = cv2.resize(mask_cropped_right, (image.shape[1], image.shape[0]))
    img_cropped_left = cv2.resize(img_cropped_left, (image.shape[1], image.shape[0]))
    img_cropped_right = cv2.resize(img_cropped_right, (image.shape[1], image.shape[0]))


    image[img_cropped_left>0] = img_cropped_left[img_cropped_left>0]
    image[img_cropped_right>0] = img_cropped_right[img_cropped_right>0]
    
    mask[mask_cropped_left>0] = mask_cropped_left[mask_cropped_left>0]
    mask[mask_cropped_right>0] = mask_cropped_right[mask_cropped_right>0]
    K = np.array([[image.shape[1]//1, 0, image.shape[1] // 2],
                [0, image.shape[1]//1, image.shape[0] // 2],
                [0, 0, 1]])
    D = np.array([0.0,10, 0.0, 0.0])
    # D = np.array([0, 0, 0.0, 0.0])
    # D = np.array([0.17149, -0.27191, 0.25787, -0.08054])


    # fish-eye 변환 행렬 계산

    fish_eye_image = cv2.undistort(image, K, D)
    fish_eye_image = fish_eye_image[130:130+750, 300:300+1460]

    fish_eye_mask = cv2.undistort(mask, K, D)
    fish_eye_mask = fish_eye_mask[130:130+750, 300:300+1460]
    return fish_eye_image, fish_eye_mask

# 이미지 로드
image = cv2.imread('./dataset/train_source_image/TRAIN_SOURCE_0000.png')
mask = cv2.imread('./dataset/train_source_gt/TRAIN_SOURCE_0000.png')
# image = cv2.resize(image, (512, 512))
# mask = cv2.resize(mask, (512, 512))
# 변환 적용

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
f_i, f_m = apply(image, mask)
# 결과 이미지 저장
cv2.imwrite('fish_eye_transformed_image.jpg', f_i)
cv2.imwrite('fish_eye_transformed_mask.jpg', f_m)
