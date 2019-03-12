import cv2
import numpy as np
import os
from glob import glob

def m4_image_save_cv(images, savepath, rows=4, zero_mean=True):
    # introduction: a series of images save as a picture
    # image: 4 dims
    # rows: how many images in a row
    # cols: how many images in a col
    # zero_mean:

    if zero_mean:
        images = images * 127.5 + 127.5
    if images.dtype != np.uint8:
        images = images.astype(np.uint8)
    img_num, img_height, img_width, nc = images.shape
    h_nums = rows
    w_nums = img_num // h_nums
    merge_image_height = h_nums * img_height
    merge_image_width = w_nums * img_width
    merge_image = np.ones([merge_image_height, merge_image_width, nc], dtype=np.uint8)
    for i in range(h_nums):
        for j in range(w_nums):
            merge_image[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width] = images[
                i * w_nums + j]

    merge_image = cv2.cvtColor(merge_image, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
    cv2.imwrite(savepath, merge_image)







