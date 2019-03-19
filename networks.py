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
        self.class_names = self.m4_get_classes(self.cfg.class_path)
        self.anchors = self.m4_get_anchors(self.cfg.achorfile_path)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)


    def build_model(self, images, bbox_true_13, bbox_true_26, bbox_true_52, input_image_shape):
        self.m4_draw_box(images, bbox_true_13, bbox_true_26, bbox_true_52, 'input_box')
        self.lr_ = tf.train.exponential_decay(self.cfg.lr, self.global_step, decay_steps=2000, decay_rate=0.8)

        model = yolo(self.cfg.norm_epsilon, self.cfg.norm_decay, self.cfg.achorfile_path, self.cfg.class_path, False)
        bbox_true = [bbox_true_13, bbox_true_26, bbox_true_52]
        output = model.yolo_inference(images, self.cfg.num_anchors / 3, self.cfg.num_classes, self.cfg.is_train)
        # print(output[0].get_shape().as_list())

        if self.cfg.is_train:

            self.loss = model.yolo_loss(self.output, bbox_true, model.anchors, self.cfg.num_classes,
                                        self.cfg.ignore_thresh,self.cfg.is_train)
            l2_loss = tf.losses.get_regularization_loss()
            self.loss += l2_loss

            loss_sum = tf.summary.scalar('loss', self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_)
            self.optim = optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            input_shape = [416.0, 416.0]
            grid_shapes = [tf.cast(tf.shape(output[l])[1:3], tf.float32) for l in range(3)]

            self.pred_xy_list = []
            self.pred_wh_list = []
            self.box_confidence_list = []
            self.box_class_probs_list = []
            y_true = [bbox_true_13, bbox_true_26, bbox_true_52]
            for index in range(3):
                # 只有负责预测ground truth box的grid对应的为1, 才计算相对应的loss
                # object_mask的shape为[batch_size, grid_size, grid_size, 3, 1]
                object_mask = y_true[index][..., 4:5]
                class_probs = y_true[index][..., 5:]
                pred_xy, pred_wh, box_confidence, box_class_probs = self.yolo_head(output[index],self.anchors[anchor_mask[index]],
                                                                     self.cfg.num_classes,input_shape,self.cfg.is_train)
                self.pred_xy_list.append(pred_xy)
                self.pred_wh_list.append(pred_wh)
                self.box_confidence_list.append(box_confidence)
                self.box_class_probs_list.append(box_class_probs)



            # self.boxes, self.scores, self.classes = self.eval(output, input_image_shape, max_boxes=20)
            # print(self.boxes.get_shape().as_list())


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

    def eval(self, yolo_outputs, image_shape, max_boxes = 20):
        """
        Introduction
        ------------
            根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
        Parameters
        ----------
            yolo_outputs: yolo模型输出
            image_shape: 图片的大小
            max_boxes:  最大box数量
        Returns
        -------
            boxes_: 物体框的位置
            scores_: 物体类别的概率
            classes_: 物体类别
        """
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = tf.shape(yolo_outputs[0])[1 : 3] * 32
        # 对三个尺度的输出获取每个预测box坐标和box的分数，score计算为置信度x类别概率
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]], len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes, axis = 0)
        box_scores = tf.concat(box_scores, axis = 0)

        mask = box_scores >= self.cfg.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype = tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(len(self.class_names)):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold = self.cfg.nms_threshold)
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis = 0)
        scores_ = tf.concat(scores_, axis = 0)
        classes_ = tf.concat(classes_, axis = 0)

        return boxes_, scores_, classes_


    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """
        Introduction
        ------------
            将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
        Parameters
        ----------
            feats: yolo输出的feature map
            anchors: anchor的位置
            class_num: 类别数目
            input_shape: 输入大小
            image_shape: 图片大小
        Returns
        -------
            boxes: 物体框的位置
            boxes_scores: 物体框的分数，为置信度和类别概率的乘积
        """
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        Introduction
        ------------
            计算物体框预测坐标在原图中的位置坐标
        Parameters
        ----------
            box_xy: 物体框左上角坐标
            box_wh: 物体框的宽高
            input_shape: 输入的大小
            image_shape: 图片的大小
        Returns
        -------
            boxes: 物体框的位置
        """
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.cast(input_shape, dtype = tf.float32)
        image_shape = tf.cast(image_shape, dtype = tf.float32)
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis = -1)
        boxes *= tf.concat([image_shape, image_shape], axis = -1)
        return boxes



    def _get_feats(self, feats, anchors, num_classes, input_shape):
        """
        Introduction
        ------------
            根据yolo最后一层的输出确定bounding box
        Parameters
        ----------
            feats: yolo模型最后一层输出
            anchors: anchors的位置
            num_classes: 类别数量
            input_shape: 输入大小
        Returns
        -------
            box_xy, box_wh, box_confidence, box_class_probs
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis = -1)
        grid = tf.cast(grid, tf.float32)
        # 将x,y坐标归一化为占416的比例
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 将w,h也归一化为占416的比例
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    def m4_get_anchors(self, filepath):
        '''
        :param filepath: 含路径的文件名称
        :return: np.arrary 的数组 (9,2)
        '''
        with open(filepath) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        achors_np = np.array(anchors).reshape(-1, 2)
        return achors_np

    def m4_get_classes(self, filepath):
        '''
        introduction: 将存储类别名称的.txt文件读出成list：['person', 'bicycle', 'car', 'motorbike', ....]
        :param filepath:含路径的文件名称
        :return:类别名称的列表， list
        '''
        with open(filepath) as f:
            class_names = f.readlines()
        class_names = [tf.constant(c.strip()) for c in class_names]
        return class_names

    def yolo_head(self, feats, anchors, num_classes, input_shape, training=True):
        """
        Introduction
        ------------
            根据不同大小的feature map做多尺度的检测，三种feature map大小分别为13x13x1024, 26x26x512, 52x52x256
        Parameters
        ----------
            feats: 输入的特征feature map
            anchors: 针对不同大小的feature map的anchor
            num_classes: 类别的数量
            input_shape: 图像的输入大小，一般为416
            trainging: 是否训练，用来控制返回不同的值
        Returns
        -------
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3] # tf.shape获取的是维度
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)
        # 将x,y坐标归一化为占416的比例
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 将w,h也归一化为占416的比例
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / input_shape[::-1]
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        if training == True:
            return grid, predictions, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs






