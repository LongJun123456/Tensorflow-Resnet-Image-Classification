import os
import cv2
import numpy as np
import random

def warpAffine_image_0(img):
    h, w = img.shape[:2]
    point1=np.float32([[50,50],[300,50],[50,200]])
    point2=np.float32([[20,70],[300,50],[80,230]])
    M=cv2.getAffineTransform(point1,point2)
    dst=cv2.warpAffine(img,M,(w,h),borderValue=(255,255,255))
    return dst

def warpAffine_image_1(img):
    h, w = img.shape[:2]
    point1=np.float32([[50,50],[300,50],[50,200]])
    point2=np.float32([[20,70],[300,50],[80,230]])
    M=cv2.getAffineTransform(point2,point1)
    dst=cv2.warpAffine(img,M,(w,h),borderValue=(255,255,255))
    #cv2.imshow("1",dst)
    #cv2.waitKey(0)
    return dst
def gamma_trans(img,gamma):
    #具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    #实现映射用的是Opencv的查表函数
    return cv2.LUT(img,gamma_table)

def gasuss_noise(image, mean=0, var=0.25):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.0
    else:
        low_clip = 0.0
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

# img = cv2.imread('5257596cde98bd441b632ccdc77617b162ef9a07.jpg')
# gama = gamma_trans(img, 0.5)
# cv2.imshow('source image', img)
# cv2.imshow('warp_1', gama)
# cv2.waitKey(0)

data_path = os.path.join('data','train')
all_dir = os.listdir(data_path)
print(all_dir)
for dir in all_dir:
    dir_path = os.path.join(data_path, dir)
    img_list = os.listdir(dir_path)
    for img_name in img_list:
        img_name_split = img_name.split('.')
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        if random.random() > 0.5:
            gasuss_noise_img = gasuss_noise(img)
            save_path = os.path.join(dir_path, img_name_split[0]+'_gasuss_noise'+'.jpg')
            cv2.imwrite(save_path, gasuss_noise_img)
        if img.shape[0] > 130 and img.shape[1] > 130:
            if random.random() > 0.5:
                warp_image_0 = warpAffine_image_0(img)
                save_path = os.path.join(dir_path, img_name_split[0]+'_warp_0'+'.jpg')
                cv2.imwrite(save_path, warp_image_0)
            if random.random() > 0.5:
                warp_image_1 = warpAffine_image_1(img)
                save_path = os.path.join(dir_path, img_name_split[0]+'_warp_1'+'.jpg')
                cv2.imwrite(save_path, warp_image_1)            


