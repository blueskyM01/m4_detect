import tensorflow as tf
import numpy as np
import os
from ops import *
from utils import *
import time
import importlib
from yolo3_model import yolo

#-----------------------------m4_BE_GAN_network-----------------------------
#---------------------------------------------------------------------------
slim = tf.contrib.slim
class m4_yolo_network:
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg

        self.global_step = tf.Variable(0, name='global_step', trainable=False)


    def build_model(self, images, bbox_true_13, bbox_true_26, bbox_true_52):
        self.m4_draw_box(images, bbox_true_13, bbox_true_26, bbox_true_52, 'input_box')
        self.lr_ = tf.train.exponential_decay(self.cfg.lr, self.global_step, decay_steps=2000, decay_rate=0.8)

        model = yolo(self.cfg.norm_epsilon, self.cfg.norm_decay, self.cfg.achorfile_path, self.cfg.class_path, False)
        bbox_true = [bbox_true_13, bbox_true_26, bbox_true_52]
        output = model.yolo_inference(images, self.cfg.num_anchors / 3, self.cfg.num_classes, self.cfg.is_train)
        self.loss = model.yolo_loss(output, bbox_true, model.anchors, self.cfg.num_classes, self.cfg.ignore_thresh)
        l2_loss = tf.losses.get_regularization_loss()
        self.loss += l2_loss

        # image_fake_sum = tf.summary.image('image_fake', self.G, 3)
        # image_real_sum = tf.summary.image('image_real', images, 3)
        loss_sum = tf.summary.scalar('loss', self.loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_)
        self.optim = optimizer.minimize(self.loss, global_step=self.global_step)




    def m4_get_true_boxes_available(self, bbox_true_13, bbox_true_26, bbox_true_52):
        '''
        :param bbox_true_13:
        :param bbox_true_26:
        :param bbox_true_52:
        :return:
        '''
        batch_boxes_13, batch_boxes_26, batch_boxes_52 = [], [], []
        for label_13, label_26, label_52 in zip(bbox_true_13, bbox_true_26, bbox_true_52):
            boxes_13, boxes_26, boxes_52 = [], [], []
            for y_idx in range(label_13.shape[0]):
                for x_idx in range(label_13.shape[1]):
                    for anchor_idx in range(label_13.shape[2]):
                        if label_13[y_idx][x_idx][anchor_idx][4] > 0:
                            y_center = label_13[y_idx][x_idx][anchor_idx][1]
                            x_center = label_13[y_idx][x_idx][anchor_idx][0]
                            y_min = y_center - label_13[y_idx][x_idx][anchor_idx][3] / 2.0
                            x_min = x_center - label_13[y_idx][x_idx][anchor_idx][2] / 2.0
                            y_max = y_center + label_13[y_idx][x_idx][anchor_idx][3] / 2.0
                            x_max = x_center + label_13[y_idx][x_idx][anchor_idx][2] / 2.0
                            cat = np.argmax(label_13[y_idx][x_idx][anchor_idx][5:])
                            box_13 = [y_min, x_min, y_max, x_max, cat]
                            boxes_13.append(box_13)

            for i in range(100 - len(boxes_13)):
                boxes_13.append([0., 0., 0., 0., 0.])
            batch_boxes_13.append(boxes_13)


            for y_idx in range(label_26.shape[0]):
                for x_idx in range(label_26.shape[1]):
                    for anchor_idx in range(label_26.shape[2]):
                        if label_26[y_idx][x_idx][anchor_idx][4] > 0:
                            y_center = label_26[y_idx][x_idx][anchor_idx][1]
                            x_center = label_26[y_idx][x_idx][anchor_idx][0]
                            y_min = y_center - label_26[y_idx][x_idx][anchor_idx][3] / 2.0
                            x_min = x_center - label_26[y_idx][x_idx][anchor_idx][2] / 2.0
                            y_max = y_center + label_26[y_idx][x_idx][anchor_idx][3] / 2.0
                            x_max = x_center + label_26[y_idx][x_idx][anchor_idx][2] / 2.0
                            cat = np.argmax(label_26[y_idx][x_idx][anchor_idx][5:])
                            box_26 = [y_min, x_min, y_max, x_max, cat]
                            boxes_26.append(box_26)

            for i in range(100 - len(boxes_26)):
                boxes_26.append([0., 0., 0., 0., 0.])
            batch_boxes_26.append(boxes_26)

            for y_idx in range(label_52.shape[0]):
                for x_idx in range(label_52.shape[1]):
                    for anchor_idx in range(label_52.shape[2]):
                        if label_52[y_idx][x_idx][anchor_idx][4] > 0:
                            y_center = label_52[y_idx][x_idx][anchor_idx][1] * 416.0
                            x_center = label_52[y_idx][x_idx][anchor_idx][0] * 416.0
                            y_min = y_center - label_52[y_idx][x_idx][anchor_idx][3] * 416.0 / 2.0
                            x_min = x_center - label_52[y_idx][x_idx][anchor_idx][2] * 416.0 / 2.0
                            y_max = y_center + label_52[y_idx][x_idx][anchor_idx][3] * 416.0 / 2.0
                            x_max = x_center + label_52[y_idx][x_idx][anchor_idx][2] * 416.0 / 2.0
                            cat = np.argmax(label_52[y_idx][x_idx][anchor_idx][5:])
                            box_52 = [y_min, x_min, y_max, x_max, cat]
                            boxes_52.append(box_52)

            for i in range(100 - len(boxes_52)):
                boxes_52.append([0., 0., 0., 0., 0.])
            batch_boxes_52.append(boxes_52)

        return np.array(batch_boxes_13,dtype=np.float32), np.array(batch_boxes_26,dtype=np.float32), np.array(batch_boxes_52,dtype=np.float32)

    def m4_draw_box(self,image, bbox_true_13, bbox_true_26, bbox_true_52, box_name):
        with tf.variable_scope('draw_bounding_box'):
            batch_boxes_13, batch_boxes_26, batch_boxes_52 = tf.py_func(self.m4_get_true_boxes_available,
                                                                     [bbox_true_13, bbox_true_26, bbox_true_52],
                                                                     [tf.float32, tf.float32, tf.float32])

            self.boxes_sum = tf.concat([batch_boxes_13, batch_boxes_26, batch_boxes_52],axis=1)
            input_image = tf.image.draw_bounding_boxes(image, self.boxes_sum)
            tf.summary.image(box_name, input_image)

    def Delete_zero_boxes(self, boxes_data_np, boxes_data_np1, boxes_data_np2):
        '''
        Introduction: py_func的应用， 用于删除插入的0box，保留可用的
        :param boxes_data_np:
        :return:
        '''
        value_box_idx = boxes_data_np[:, 0] > 0
        boxes_data_np_available = boxes_data_np[value_box_idx]

        value_box_idx1 = boxes_data_np1[:, 0] > 0
        boxes_data_np_available1 = boxes_data_np1[value_box_idx1]

        value_box_idx2 = boxes_data_np2[:, 0] > 0
        boxes_data_np_available2 = boxes_data_np2[value_box_idx2]


        return boxes_data_np_available, boxes_data_np_available1, boxes_data_np_available2









