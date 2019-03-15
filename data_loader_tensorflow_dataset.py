import tensorflow as tf
import numpy as np
import os
from collections import defaultdict
import json
import cv2
import time


class m4_ReadData:
    def __init__(self, is_train, dataset_dir, dataset_name, label_dir, label_name,
                 anchors_path, class_path, num_classes, max_boxes=20, input_shape=416,
                 batch_size=16, epoch=30, buffer_size=10000):
        '''
        :param mode:
        :param dataset_dir:
        :param dataset_name:
        :param label_dir:
        :param label_name:
        :param anchors_path:
        :param class_path:
        :param num_classes:
        :param max_boxes:
        :param input_shape:
        :param batch_size:
        :param epoch:
        :param buffer_size:
        '''
        self.is_train = is_train
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.label_dir = label_dir
        self.label_name = label_name
        self.anchors_path = anchors_path
        self.class_path = class_path
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epoch = epoch
        self.buffer_size = buffer_size
        self.class_name = self.m4_get_classes(self.class_path)
        self.anchors = self.m4_get_anchors(self.anchors_path)
        class_names = self.m4_get_classes(self.class_path)



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
        class_names = [c.strip() for c in class_names]
        return class_names

    def m4_read_annotations(self, dataset_dir, dataset_name, label_dir, label_name):
        '''
        Introduction: 读取COCO数据集图片路径和对应的标注
        :param label_dir:
        :param label_name:
        ------------------------------------------
        :return: COCO数据集图片路径list, box的list
        image_data: 存有图像路径名称的列表['name1','name2',....]
        boxes_data： 也是个列表  format：[[np.(xmin,ymin,xmax,ymax,cat),np.(xmin,ymin,xmax,ymax,cat)...],....]
        image_data和boxes_data中的原始一一对应
        '''
        image_data = []
        boxes_data = []
        name_box_id = defaultdict(list)
        with open(os.path.join(label_dir, label_name), encoding='utf-8') as file:
            data = json.load(file)
            annotations = data['annotations']
            for ant in annotations:
                id = ant['image_id']
                name = os.path.join(dataset_dir,dataset_name, '%012d.jpg' % id)
                cat = ant['category_id']
                if cat >= 1 and cat <= 11:
                    cat = cat - 1
                elif cat >= 13 and cat <= 25:
                    cat = cat - 2
                elif cat >= 27 and cat <= 28:
                    cat = cat - 3
                elif cat >= 31 and cat <= 44:
                    cat = cat - 5
                elif cat >= 46 and cat <= 65:
                    cat = cat - 6
                elif cat == 67:
                    cat = cat - 7
                elif cat == 70:
                    cat = cat - 9
                elif cat >= 72 and cat <= 82:
                    cat = cat - 10
                elif cat >= 84 and cat <= 90:
                    cat = cat - 11

                name_box_id[name].append([ant['bbox'], cat])

            for key in name_box_id.keys():
                boxes = []
                image_data.append(key)
                box_infos = name_box_id[key]
                for info in box_infos:
                    x_min = info[0][0]
                    y_min = info[0][1]
                    x_max = x_min + info[0][2]
                    y_max = y_min + info[0][3]
                    boxes.append(np.array([x_min, y_min, x_max, y_max, info[1]]))
                boxes_data.append(np.array(boxes))

        return image_data, boxes_data

    def m4_convertTo_Tensor(self,image_data, boxes_data):
        '''
        Introduction: 将获得的图像名称列表， boxes的列表转换成tensor
        :param image_data:
        :param boxes_data:
        :return: 对应的tensor
        '''
        add_element = np.zeros(shape=[1,5], dtype=np.float32)
        # 将每张图像中的box统一到100，为了转成tensor
        for idx in range(len(boxes_data)):
            if boxes_data[idx].shape[0] < 100:
                for ii in range(100 - boxes_data[idx].shape[0]):
                    boxes_data[idx] = np.append(boxes_data[idx], add_element,axis=0)
        image_data_tensor = tf.constant(image_data)
        boxes_data_tensor = tf.convert_to_tensor(np.array(boxes_data).astype(np.float32),dtype=tf.float32)
        return image_data_tensor, boxes_data_tensor

    def m4_Preprocess(self, image, bbox):
        """
        Introduction
        ------------
            对图片进行预处理，增强数据集
        Parameters
        ----------
            image: tensorflow解析的图片
            bbox: 图片中对应的box坐标
        """
        # 转换成tf.float32型
        image_width, image_high = tf.cast(tf.shape(image)[1], tf.float32), tf.cast(tf.shape(image)[0], tf.float32)
        input_width = tf.cast(self.input_shape, tf.float32)
        input_high = tf.cast(self.input_shape, tf.float32)
        # 放缩图像， 放缩后的图像的窗框不超过指定输入的尺寸，即416
        new_high = image_high * tf.minimum(input_width / image_width, input_high / image_high)
        new_width = image_width * tf.minimum(input_width / image_width, input_high / image_high)
        # 将图片按照固定长宽比进行padding缩放
        dx = (input_width - new_width) / 2
        dy = (input_high - new_high) / 2
        # 将图像放缩到新的new_high，new_width
        image = tf.image.resize_images(image, [tf.cast(new_high, tf.int32), tf.cast(new_width, tf.int32)],
                                       method = tf.image.ResizeMethod.BICUBIC)
        # 将图像填充到 input_width， input_high
        new_image = tf.image.pad_to_bounding_box(image,
                                                 tf.cast(dy, tf.int32), tf.cast(dx, tf.int32),
                                                 tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))
        image_ones = tf.ones_like(image)
        image_ones_padded = tf.image.pad_to_bounding_box(image_ones,
                                                         tf.cast(dy, tf.int32), tf.cast(dx, tf.int32),
                                                         tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))
        image_color_padded = (1 - image_ones_padded) * 0.5 # 图像已经归一到0-1之间， 因此中间颜色是0.5
        image = image_color_padded + new_image  # 将图像填充到 input_width， input_high后， 边上填充平均颜色0.5
        # 矫正bbox坐标
        xmin, ymin, xmax, ymax, label = tf.split(value = bbox, num_or_size_splits=5, axis = 1) # 维度不变
        xmin = xmin * new_width / image_width + dx
        xmax = xmax * new_width / image_width + dx
        ymin = ymin * new_high / image_high + dy
        ymax = ymax * new_high / image_high + dy
        bbox = tf.concat([xmin, ymin, xmax, ymax, label], 1)
        # if self.mode == 'train':
        #     # 随机左右翻转图片
        #     def _flip_left_right_boxes(boxes):
        #         xmin, ymin, xmax, ymax, label = tf.split(value = boxes, num_or_size_splits = 5, axis = 1)
        #         flipped_xmin = tf.subtract(input_width, xmax)
        #         flipped_xmax = tf.subtract(input_width, xmin)
        #         flipped_boxes = tf.concat([flipped_xmin, ymin, flipped_xmax, ymax, label], 1)
        #         return flipped_boxes
        #     flip_left_right = tf.greater(tf.random_uniform([], dtype = tf.float32, minval = 0, maxval = 1), 0.5)
        #     image = tf.cond(flip_left_right, lambda : tf.image.flip_left_right(image), lambda : image)
        #     bbox = tf.cond(flip_left_right, lambda: _flip_left_right_boxes(bbox), lambda: bbox)
        # 将图片归一化到0和1之间
        # image = image / 255. # 已经是0-1
        image = tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)
        bbox = tf.clip_by_value(bbox, clip_value_min = 0, clip_value_max = input_width - 1)
        # batch取的shape要一致， 因此都统一到max_boxes的数量
        bbox = tf.cond(tf.greater(tf.shape(bbox)[0], self.max_boxes), lambda: bbox[:self.max_boxes], lambda: tf.pad(bbox, paddings = [[0, self.max_boxes - tf.shape(bbox)[0]], [0, 0]], mode = 'CONSTANT'))
        return image, bbox

    def Preprocess_true_boxes(self, true_boxes_):
        """
        Introduction
        ------------
            对训练数据的ground truth box进行预处理
            tf.py_func,处理的是numpy
        Parameters
        ----------
            true_boxes: ground truth box 形状为[boxes, 5], x_min, y_min, x_max, y_max, class_id
        """
        true_boxes = true_boxes_.copy()
        num_layers = len(self.anchors) // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array([self.input_shape, self.input_shape], dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2. # box的中心坐标
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2] # box的w,h
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1] # 中心坐标按输入图像的长宽做归一 input_shape[::-1]倒序
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1] # 长宽按输入图像的长宽做归一
        grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8] # 取13x13, 26x26, 52x52的网格，每个网格的长宽
        y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.num_classes), dtype='float32')
                  for l in range(num_layers)]

        # 这里扩充维度是为了后面应用广播计算每个图中所有box的anchor互相之间的iou
        anchors = np.expand_dims(self.anchors, 0) # (1,9,2)
        anchors_max = anchors / 2. # 元素变为一半
        anchors_min = -anchors_max # 元素变为一半，加个'-'， 就是零均值化   (1,9,2)
        # 因为之前对box做了padding, 因此需要去除全0行
        valid_mask = boxes_wh[..., 0] > 0
        wh = boxes_wh[valid_mask] # (box_num, 2)
        # 为了应用广播扩充维度
        wh = np.expand_dims(wh, -2) # wh 的shape为[box_num, 1, 2]
        boxes_max = wh / 2.
        boxes_min = -boxes_max

        intersect_min = np.maximum(boxes_min, anchors_min)
        intersect_max = np.minimum(boxes_max, anchors_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area) # (box_nums,9)

        # 找出和ground truth box的iou最大的anchor box, 然后将对应不同比例的负责该ground turth box 的位置置为ground truth box坐标
        best_anchor = np.argmax(iou, axis = -1)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[t, 4].astype('int32')
                    y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                    y_true[l][j, i, k, 4] = 1.
                    y_true[l][j, i, k, 5 + c] = 1.
        return y_true[0], y_true[1], y_true[2]

    def Delete_zero_boxes(self, boxes_data_np):
        '''
        Introduction: py_func的应用， 用于删除插入的0box，保留可用的
        :param boxes_data_np:
        :return:
        '''
        value_box_idx = boxes_data_np[:, 0] > 0
        boxes_data_np_available = boxes_data_np[value_box_idx]
        return boxes_data_np_available


    def m4_parse_function(self, image_data_tensor, boxes_data_tensor):
        # 删除插入的0box，保留可用的
        [boxes_data_tensor_available] = tf.py_func(self.Delete_zero_boxes, [boxes_data_tensor],[tf.float32])
        # 读取原始图片，范围0-1
        image_string = tf.read_file(image_data_tensor)
        image_decoded = tf.image.decode_jpeg(image_string, 3)
        image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32) # 0-1
        # 预处理图像
        pre_image, pre_boxes = self.m4_Preprocess(image_decoded, boxes_data_tensor_available) # input_shape x input_shape

        # make label of yolo
        bbox_true_13, bbox_true_26, bbox_true_52 = tf.py_func(self.Preprocess_true_boxes, [pre_boxes],
                                                              [tf.float32, tf.float32, tf.float32])

        return pre_image, pre_boxes, bbox_true_13, bbox_true_26, bbox_true_52


    def data_loader(self):
        '''
        :param is_train:
        :param dataset_dir:
        :param dataset_name:
        :param label_dir:
        :param label_name:
        :param anchors_path:
        :param class_path:
        :param num_classes:
        :param max_boxes:
        :param input_shape:
        :param batch_size:
        :param epoch:
        :param buffer_size:
        :return:
        '''
        image_data, boxes_data = self.m4_read_annotations(self.dataset_dir, self.dataset_name,
                                                          self.label_dir, self.label_name)
        dataset_size = len(image_data)
        image_data_tensor, boxes_data_tensor = self.m4_convertTo_Tensor(image_data, boxes_data)

        try:
            dataset = tf.data.Dataset.from_tensor_slices((image_data_tensor, boxes_data_tensor))
        except:
            dataset = tf.contrib.data.Dataset.from_tensor_slices((image_data_tensor, boxes_data_tensor))

        dataset = dataset.map(self.m4_parse_function)
        # dataset = dataset.shuffle(self.buffer_size=10000).batch(self.batch_size).repeat(self.epoch)

        dataset = dataset.batch(self.batch_size).repeat(self.epoch)

        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        return one_element, dataset_size


