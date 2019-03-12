import tensorflow as tf

'''
#-----------------------------m4_gan_network-----------------------------
dataset_dir = '/media/yang/F/DataSet/Face'
dataset_name = 'lfw-deepfunneled'
datalabel_dir = '/media/yang/F/DataSet/Face/Label'
datalabel_name = 'pair_FGLFW.txt'
log_dir = './logs'
sampel_save_dir = './samples'
num_gpus = 2
epoch = 40
learning_rate = 0.001
beta1 = 0.5
beta2 = 0.5
batch_size = 16
z_dim = 128
g_feats = 64
saveimage_period = 10
savemodel_period = 40
#-----------------------------m4_gan_network-----------------------------
'''

# -----------------------------m4_BE_GAN_network-----------------------------
gpu_assign = '0'
is_train = True
save_dir = '/yolo_reslut/'
dataset_dir = '/media/yang/F/DataSet/Tracking'
dataset_name = 'train2017'
datalabel_dir = '/media/yang/F/DataSet/Tracking/annotations_trainval2017/annotations'
datalabel_name = 'instances_train2017.json'
achorfile_path = './yolo_anchors.txt'
class_path = './coco_classes.txt'

log_dir = '/media/yang/F/ubuntu/SD_GAN_Result' + save_dir+'logs'  # need to change

checkpoint_dir = '/media/yang/F/ubuntu/SD_GAN_Result' + save_dir+'checkpoint'  # need to change

num_gpus = 1
epoch = 10
batch_size = 1  # need to change
num_classes = 80
max_boxes = 20
input_shape = 416



lr_lower_boundary = 0.00002
add_summary_period = 10
savemodel_period = 1
lr_drop_period = 1



