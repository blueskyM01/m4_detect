import tensorflow as tf
import numpy as np
import os
import datetime
from utils import *
from ops import *
from networks import *
import time
import scipy
import scipy.io as sio
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第  块GPU可用
# print(image_data[0])
# print(boxes_data[0])
# img = cv2.imread(image_data[0])
# font = cv2.FONT_HERSHEY_SIMPLEX
# for idx in range(boxes_data[0].shape[0]):
#     tl = (int(boxes_data[0][idx][0]), int(boxes_data[0][idx][1]))
#     br = (int(boxes_data[0][idx][2]), int(boxes_data[0][idx][3]))
#     cv2.putText(img, self.class_names[int(boxes_data[0][idx][4])], tl, font, 0.5, (255, 0, 0), 1)
#     cv2.rectangle(img, tl, br, (0, 0, 255), 2)
#
# cv2.imshow('show', img)
# cv2.waitKey(0)

# img = image_decoded[0].astype(np.uint8)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # cv2默认为bgr顺序
# for idx in range(boxes_data_tensor_available[0].shape[0]):
#     tl = (int(boxes_data_tensor_available[0][idx][0]), int(boxes_data_tensor_available[0][idx][1]))
#     br = (int(boxes_data_tensor_available[0][idx][2]), int(boxes_data_tensor_available[0][idx][3]))
#     # cv2.putText(image_decoded[0], self.class_names[int(boxes_data_tensor_available[0][idx][4])], tl, font, 0.5, (255, 0, 0), 1)
#     cv2.rectangle(img, tl, br, (0, 0, 255), 2)
#
# cv2.imshow('show', img)
# cv2.waitKey(0)
#
# print(image_decoded[0])
# print(boxes_data_tensor_available[0])

npll = np.array([[-1, 2],
                 [2, 4]])
npgg = np.array([[4, 9],
                 [77, 44],
                 [-9, 5]])
for i in npgg:
    print(i)