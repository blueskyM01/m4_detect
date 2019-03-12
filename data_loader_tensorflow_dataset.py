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

        anchors = self.m4_get_anchors(self.anchors_path)
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

    def m4_Preprocess(self, image_data_tensor, boxes_data_tensor):

        '''

        :param image_data_tensor:
        :param boxes_data_tensor:
        :return:
        '''
        # 删除多的加入的box
        [boxes_data_tensor] = tf.py_func(self.Delete_zero_boxes, [boxes_data_tensor], [tf.float32])

    def Delete_zero_boxes(self, boxes_data_np):
        value_box_idx = boxes_data_np[:, 0] > 0
        boxes_data_np_available = boxes_data_np[value_box_idx]
        return boxes_data_np_available


    def m4_parse_function(self, image_data_tensor, boxes_data_tensor):
        [boxes_data_tensor_available] = tf.py_func(self.Delete_zero_boxes, [boxes_data_tensor],[tf.float32])
        image_string = tf.read_file(image_data_tensor)
        image_decoded = tf.image.decode_jpeg(image_string, 3)
        image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32) * 255.0
        # image_resized = tf.image.resize_images(image_decoded, [128, 128])
        return image_decoded, boxes_data_tensor_available


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



    # def data_loader(self, mode, dataset_dir, dataset_label_dir, batch_size=16, epoch=30, buffer_size=10000):
    #     '''
    #     :param mode:
    #     :param dataset_dir:
    #     :param dataset_label_dir:
    #     :param batch_size:
    #     :param epoch:
    #     :param buffer_size:
    #     :return:
    #     '''
    #
    #     names = np.loadtxt(os.path.join(label_dir, label_name), dtype=np.str)
    #     dataset_size = names.shape[0]
    #     names, labels = m4_get_file_label_name(label_dir, label_name, dataset_dir, dataset_name)
    #     filenames = tf.constant(names)
    #     filelabels = tf.constant(labels)
    #     try:
    #         dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
    #     except:
    #         dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, filelabels))
    #
    #     dataset = dataset.map(m4_parse_function)
    #     dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).repeat(epoch)
    #     iterator = dataset.make_one_shot_iterator()
    #     one_element = iterator.get_next()
    #     return one_element, dataset_size


# dataset_dir = '/media/yang/F/DataSet/Tracking'
# dataset_name= 'train2017'
# label_dir = '/media/yang/F/DataSet/Tracking/annotations_trainval2017/annotations'
# label_name = 'instances_train2017.json'
# achorfile_path = './yolo_anchors.txt'
# class_path = './coco_classes.txt'
# m4_read = m4_ReadData(is_train=True, dataset_dir=dataset_dir, dataset_name=dataset_name, label_dir=label_dir, label_name=label_name,
#                  anchors_path=achorfile_path, class_path=class_path, num_classes=80, max_boxes=20, input_shape=416,
#                  batch_size=16, epoch=30, buffer_size=10000)
#
#
#
#
# m4_anchors = m4_read.anchors
# m4_classes = m4_read.class_names
# print(m4_anchors.shape)
# print(m4_classes)





def m4_get_file_label_name(label_dir,label_name,dataset_dir,dataset_name):
    '''
    :param label_dir: label dir
    :param label_name: label name
    :param dataset_dir: dataset dir
    :param dataset_name: dataset name
    :return:filename_list, label_list
    '''
    filepath_name = os.path.join(label_dir,label_name)
    save_data_path_name = os.path.join(dataset_dir,dataset_name)
    data = np.loadtxt(filepath_name,dtype=str)
    filename = data[:,0].tolist()
    label=data[:,1].tolist()
    filename_list = []
    label_list=[]
    for i in range(data.shape[0]):
        filename_list.append(os.path.join(save_data_path_name,filename[i].lstrip("b'").rstrip("'")))
        label_list.append(int(label[i].lstrip("b'").rstrip("'")))
    return filename_list,label_list


