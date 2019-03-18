from __future__ import division, print_function, absolute_import
import os
import argparse
import tensorflow as tf
import param
from model import my_yolo3
import time



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
# TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
# TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
# TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
# TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息


parser = argparse.ArgumentParser()

# -----------------------------m4_BE_GAN_network-----------------------------
parser.add_argument("--gpu_assign", default=param.gpu_assign, type=str, help="assign gpu")
parser.add_argument("--is_train", default=param.is_train, type=bool, help="Train")
parser.add_argument("--dataset_dir", default=param.dataset_dir, type=str, help="Train data set dir")
parser.add_argument("--dataset_name", default=param.dataset_name, type=str, help="Train data set name")
parser.add_argument("--datalabel_dir", default=param.datalabel_dir, type=str, help="Train data label dir")
parser.add_argument("--datalabel_name", default=param.datalabel_name, type=str, help="Train data label name")
parser.add_argument("--achorfile_path", default=param.achorfile_path, type=str, help="achor file path")
parser.add_argument("--class_path", default=param.class_path, type=str, help="class file path")


parser.add_argument("--log_dir", default=param.log_dir, type=str, help="Train data label name")
parser.add_argument("--checkpoint_dir", default=param.checkpoint_dir, type=str, help="model save dir")
parser.add_argument("--num_gpus", default=param.num_gpus, type=int, help="num of gpu")
parser.add_argument("--epoch", default=param.epoch, type=int, help="epoch")
parser.add_argument("--batch_size", default=param.batch_size, type=int, help="batch size for one gpus")
parser.add_argument("--num_classes", default=param.num_classes, type=int, help="number of classes")
parser.add_argument("--num_anchors", default=param.num_anchors, type=int, help="number of anchors")

parser.add_argument("--max_boxes", default=param.max_boxes, type=int, help="max boxes")
parser.add_argument("--input_shape", default=param.input_shape, type=int, help="the width and height of image are all input_shape")

parser.add_argument("--lr_lower_boundary", default=param.lr_lower_boundary, type=float, help="lower learning rate")
parser.add_argument("--norm_epsilon", default=param.norm_epsilon, type=float, help="norm_epsilon")
parser.add_argument("--norm_decay", default=param.norm_decay, type=float, help="norm_decay")
parser.add_argument("--lr", default=param.lr, type=float, help="learning rate")
parser.add_argument("--ignore_thresh", default=param.ignore_thresh, type=float, help="ignore thresh")



parser.add_argument("--savemodel_period", default=param.savemodel_period, type=int, help="savemodel_period")
parser.add_argument("--add_summary_period", default=param.add_summary_period, type=int, help="add_summary_period")
parser.add_argument("--lr_drop_period", default=param.lr_drop_period, type=int, help="lr_drop_period")




cfg = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_assign  # 指定第  块GPU可用

if __name__ == '__main__':

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        my_yolo3 = my_yolo3(sess, cfg)
        if cfg.is_train:
            print('Training start....')
            if not os.path.exists(cfg.log_dir):
                os.makedirs(cfg.log_dir)
            if not os.path.exists(cfg.checkpoint_dir):
                os.makedirs(cfg.checkpoint_dir)
            my_yolo3.train()
        else:
            print('test starting ....')
            time.sleep(3)
            my_yolo3.test()
