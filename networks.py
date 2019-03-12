import tensorflow as tf
import numpy as np
import os
from ops import *
from utils import *
import time
import ExpShapePoseNet as ESP
import importlib

#-----------------------------m4_BE_GAN_network-----------------------------
#---------------------------------------------------------------------------
slim = tf.contrib.slim
class m4_BE_GAN_network:
    def __init__(self, sess, cfg, new_graph):
        self.sess = sess
        self.cfg = cfg
        self.new_graph = new_graph
        # self.batch_idx = batch_idx
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.conv_hidden_num = cfg.conv_hidden_num
        self.data_format = cfg.data_format
        self.z_dim = cfg.z_dim
        self.gamma = self.cfg.gamma
        self.lambda_k = self.cfg.lambda_k
        self.g_lr = tf.Variable(self.cfg.g_lr, name='g_lr')
        self.d_lr = tf.Variable(self.cfg.d_lr, name='d_lr')

    def build_model(self, images, labels, z, images_new_graph, shape_real, pose_real, expr_real):
        _, height, width, self.channel = \
            self.get_conv_shape(images, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.9, self.cfg.lr_lower_boundary), name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.9, self.cfg.lr_lower_boundary), name='d_lr_update')
        self.k_t = tf.Variable(0., trainable=False, name='k_t')
        self.op_g = tf.train.AdamOptimizer(learning_rate=self.g_lr)
        self.op_d = tf.train.AdamOptimizer(learning_rate=self.d_lr)

        id_feat_real = self.m4_ID_Extractor(images,reuse=False)
        self.shape_real_norm, self.expr_real_norm, self.pose_real_norm, self.new_sess = self.model_3DMM_new_graph(
                                                                                        self.new_graph, images_new_graph) # get real feat
        z_concat_feat = tf.concat([z, shape_real, pose_real, expr_real, id_feat_real], axis=1)


        self.G, self.G_var = self.GeneratorCNN( z_concat_feat, self.conv_hidden_num, self.channel, self.repeat_num, self.data_format, reuse=False)
        id_feat_fake = self.m4_ID_Extractor(self.G,reuse=True)
        shape_fake_norm, expr_fake_norm, pose_fake_norm = self.model_3DMM_default_graph(self.G) # get fake feat

        self.shape_loss = tf.reduce_mean(tf.square(shape_real - shape_fake_norm))
        self.expr_loss = tf.reduce_mean(tf.square(expr_real - expr_fake_norm))
        self.pose_loss = tf.reduce_mean(tf.square(pose_real - pose_fake_norm))
        self.id_loss = tf.reduce_mean(tf.square(id_feat_real - id_feat_fake))

        d_out, self.D_z, self.D_var = self.DiscriminatorCNN(
            tf.concat([self.G, images], 0), self.channel, self.z_dim, self.repeat_num,
            self.conv_hidden_num, self.data_format)
        AE_G, AE_x = tf.split(d_out, 2)

        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - images))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_G - self.G))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.abs(AE_G - self.G)) + self.cfg.lambda_s * self.shape_loss + self.cfg.lambda_e * self.expr_loss \
                                                            + self.cfg.lambda_p * self.pose_loss + self.cfg.lambda_id * self.id_loss


        image_fake_sum = tf.summary.image('image_fake', self.G, 3)
        image_real_sum = tf.summary.image('image_real', images, 3)
        g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
        shape_loss_sum = tf.summary.scalar('shape_loss', self.shape_loss)
        expr_loss_sum = tf.summary.scalar('expr_loss', self.expr_loss)
        pose_loss_sum = tf.summary.scalar('pose_loss', self.pose_loss)
        id_loss_sum = tf.summary.scalar('id_loss', self.id_loss)

        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]

        self.g_optim = self.op_g.minimize(self.g_loss, var_list=self.g_vars)
        self.d_optim = self.op_d.minimize(self.d_loss, var_list=self.d_vars,global_step=self.global_step)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)
        self.measure_sum = tf.summary.scalar('measure', self.measure)
        with tf.control_dependencies([self.d_optim, self.g_optim]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

    def GeneratorCNN(self, z, hidden_num, output_num, repeat_num, data_format, reuse):
        with tf.variable_scope("generator", reuse=reuse) as vs:
            num_output = int(np.prod([8, 8, hidden_num]))
            x = slim.fully_connected(z, num_output, activation_fn=None)
            x = self.reshape(x, 8, 8, hidden_num, data_format)

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                if idx < repeat_num - 1:
                    x = self.upscale(x, 2, data_format)

            out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)

        variables = tf.contrib.framework.get_variables(vs)
        return out, variables

    def DiscriminatorCNN(self, x, input_channel, z_num, repeat_num, hidden_num, data_format):
        with tf.variable_scope("discriminator") as vs:
            # Encoder
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

            prev_channel_num = hidden_num
            for idx in range(repeat_num):
                channel_num = hidden_num * (idx + 1)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                if idx < repeat_num - 1:
                    x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
                    # x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

            x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
            z = x = slim.fully_connected(x, z_num, activation_fn=None)

            # Decoder
            num_output = int(np.prod([8, 8, hidden_num]))
            x = slim.fully_connected(x, num_output, activation_fn=None)
            x = self.reshape(x, 8, 8, hidden_num, data_format)

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                if idx < repeat_num - 1:
                    x = self.upscale(x, 2, data_format)

            out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

        variables = tf.contrib.framework.get_variables(vs)
        return out, z, variables

    def get_conv_shape(self,tensor, data_format):
        shape = self.int_shape(tensor)
        # always return [N, H, W, C]
        if data_format == 'NCHW':
            return [shape[0], shape[2], shape[3], shape[1]]
        elif data_format == 'NHWC':
            return shape

    def upscale(self,x, scale, data_format):
        _, h, w, _ = self.get_conv_shape(x, data_format)
        return self.resize_nearest_neighbor(x, (h * scale, w * scale), data_format)

    def int_shape(self,tensor):
        shape = tensor.get_shape().as_list()
        return [num if num is not None else -1 for num in shape]

    def reshape(self,x, h, w, c, data_format):
        if data_format == 'NCHW':
            x = tf.reshape(x, [-1, c, h, w])
        else:
            x = tf.reshape(x, [-1, h, w, c])
        return x

    def resize_nearest_neighbor(self, x, new_size, data_format):
        if data_format == 'NCHW':
            x = nchw_to_nhwc(x)
            x = tf.image.resize_nearest_neighbor(x, new_size)
            x = nhwc_to_nchw(x)
        else:
            x = tf.image.resize_nearest_neighbor(x, new_size)
        return x


    def model_3DMM_new_graph(self, new_graph, images_3DMM):
        with new_graph.as_default():
            expr_shape_pose = ESP.m4_3DMM(self.cfg)
            expr_shape_pose.extract_PSE_feats(images_3DMM)
            fc1ls_real = expr_shape_pose.fc1ls
            fc1le_real = expr_shape_pose.fc1le
            pose_model_real = expr_shape_pose.pose
            shape_norm = tf.nn.l2_normalize(fc1ls_real,dim=0)
            expr_norm = tf.nn.l2_normalize(fc1le_real,dim=0)
            pose_norm = tf.nn.l2_normalize(pose_model_real, dim=0)

            sess_3DMM = tf.Session(graph=new_graph)
            try:
                sess_3DMM.run(tf.global_variables_initializer())
            except:
                sess_3DMM.run(tf.initialize_all_variables())
            self.load_expr_shape_pose_param_new_graph(sess_3DMM)
            return shape_norm, expr_norm, pose_norm, sess_3DMM

    def model_3DMM_default_graph(self,images):
        expr_shape_pose = ESP.m4_3DMM(self.cfg)
        expr_shape_pose.extract_PSE_feats(images)
        fc1ls = expr_shape_pose.fc1ls
        fc1le = expr_shape_pose.fc1le
        pose_model = expr_shape_pose.pose
        shape_norm = tf.nn.l2_normalize(fc1ls,dim=0)
        expr_norm = tf.nn.l2_normalize(fc1le,dim=0)
        pose_norm = tf.nn.l2_normalize(pose_model, dim=0)
        return shape_norm, expr_norm, pose_norm


    def load_expr_shape_pose_param_new_graph(self, sess):
        # Add ops to save and restore all the variables.
        saver_pose = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Spatial_Transformer'))
        saver_ini_shape_net = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shapeCNN'))
        saver_ini_expr_net = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='exprCNN'))

        # Load face pose net model from Chang et al.'ICCVW17
        try:
            load_path = self.cfg.fpn_new_model_ckpt_file_path
            saver_pose.restore(sess, load_path)
            print('Load ' + self.cfg.fpn_new_model_ckpt_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.fpn_new_model_ckpt_file_path + ' failed....')

        # load 3dmm shape and texture model from Tran et al.' CVPR2017
        try:
            load_path = self.cfg.Shape_Model_file_path
            saver_ini_shape_net.restore(sess, load_path)
            print('Load ' + self.cfg.Shape_Model_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.Shape_Model_file_path + ' failed....')
        # load our expression net model
        try:
            load_path = self.cfg.Expression_Model_file_path
            saver_ini_expr_net.restore(sess, load_path)
            print('Load ' + self.cfg.Expression_Model_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.Expression_Model_file_path + ' failed....')
        time.sleep(3)

    def m4_ID_Extractor(self, images, reuse=False):
        with tf.variable_scope('facenet',reuse=reuse) as scope:
            network = importlib.import_module('inception_resnet_v1')
            prelogits, _ = network.inference(images, 1.0,
                                             phase_train=False, bottleneck_layer_size=128,
                                             weight_decay=0.0005)
            # logits = slim.fully_connected(prelogits, 10575, activation_fn=None,
            #                               weights_initializer=slim.initializers.xavier_initializer(),
            #                               weights_regularizer=slim.l2_regularizer(0.0000),
            #                               scope='Logits', reuse=reuse)

            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings') # this is we need id feat
        return embeddings