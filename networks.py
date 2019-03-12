import tensorflow as tf
import numpy as np
import os
from ops import *
from utils import *
import time
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



