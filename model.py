import tensorflow as tf
import numpy as np
import os
import datetime
from utils import *
from ops import *
from networks import *
from data_loader_tensorflow_dataset import *
import time


class my_yolo3:
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg

        self.images = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 128, 128, 3], name='real_image')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 10575], name='id')

        #
        # m4_BE_GAN_model = m4_BE_GAN_network(self.sess, self.cfg, self.new_graph)
        # m4_BE_GAN_model.build_model(self.images, self.labels, self.z, self.images_new_graph, self.shape_real, self.pose_real,
        #                             self.expr_real)

    def train(self):
        # try:
        #     self.saver = tf.train.Saver()
        # except:
        #     print('one model save error....')
        # try:
        #     tf.global_variables_initializer().run()
        # except:
        #     tf.initialize_all_variables().run()
        #
        # self.writer = tf.summary.FileWriter('{}/{}'.format(self.cfg.log_dir,
        #                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))),
        #                                                    self.sess.graph)
        # merged = tf.summary.merge_all()

        # load all train param
        # could_load, counter = self.load(self.cfg.checkpoint_dir, self.cfg.dataset_name)
        # if could_load:
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")

        m4_DataReader = m4_ReadData(is_train=self.cfg.is_train, dataset_dir=self.cfg.dataset_dir, dataset_name=self.cfg.dataset_name,
                              label_dir=self.cfg.datalabel_dir, label_name=self.cfg.datalabel_name,
                              anchors_path=self.cfg.achorfile_path, class_path=self.cfg.class_path, num_classes=self.cfg.num_classes,
                              max_boxes=self.cfg.max_boxes, input_shape=self.cfg.input_shape, batch_size=self.cfg.batch_size,
                              epoch=self.cfg.epoch, buffer_size=10000)
        one_element, dataset_size = m4_DataReader.data_loader()
        image_decoded, boxes_data_tensor_available = self.sess.run(one_element)

        img = image_decoded[0].astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # cv2默认为bgr顺序
        for idx in range(boxes_data_tensor_available[0].shape[0]):
            tl = (int(boxes_data_tensor_available[0][idx][0]), int(boxes_data_tensor_available[0][idx][1]))
            br = (int(boxes_data_tensor_available[0][idx][2]), int(boxes_data_tensor_available[0][idx][3]))
            # cv2.putText(image_decoded[0], self.class_names[int(boxes_data_tensor_available[0][idx][4])], tl, font, 0.5, (255, 0, 0), 1)
            cv2.rectangle(img, tl, br, (0, 0, 255), 2)

        cv2.imshow('show', img)
        cv2.waitKey(0)

        print(image_decoded[0])
        print(boxes_data_tensor_available[0])


        # image_data_tensor = m4_DataReader.image_data_tensor
        # boxes_data_tensor = m4_DataReader.boxes_data_tensor
        # hahah,aaaa = self.sess.run([image_data_tensor[0], boxes_data_tensor[0]])
        # print(hahah)
        # print(aaaa)

        # anchors = m4_DataReader.anchors
        # classes = m4_DataReader.class_names
        # print(anchors)
        # print(classes)



        # one_element, dataset_size = data_loader(self.cfg.datalabel_dir, self.cfg.datalabel_name, self.cfg.dataset_dir,
        #                                         self.cfg.dataset_name, self.cfg.batch_size, self.cfg.epoch)

        # batch_idxs = dataset_size // (self.cfg.batch_size)

        '''
        # try:
        for epoch in range(1,self.cfg.epoch+1):
            for idx in range(1, batch_idxs + 1):
                starttime = datetime.datetime.now()
                batch_images, batch_labels = self.sess.run(one_element)
                batch_z = np.random.uniform(-1, 1, [self.cfg.batch_size * self.cfg.num_gpus, self.cfg.z_dim]).astype(
                    np.float32)
                if batch_images.shape[0] < self.cfg.batch_size * self.cfg.num_gpus:
                    for add_idx in range(self.cfg.batch_size * self.cfg.num_gpus - batch_images.shape[0]):
                        batch_images = np.append(batch_images,batch_images[0:1],axis=0)

                (shape_real_norm, expr_real_norm, pose_real_norm) = self.new_sess.run(
                                                                    [self.shape_real_norm, self.expr_real_norm, self.pose_real_norm],
                                                                    feed_dict={self.images_new_graph: batch_images})

                # get measure stand
                k_update, k_t, Mglobal = self.sess.run([self.k_update, self.k_t, self.Mglobal],
                                                       feed_dict={self.images: batch_images,
                                                                  self.z: batch_z,
                                                                  self.shape_real:shape_real_norm,
                                                                  self.pose_real:pose_real_norm,
                                                                  self.expr_real:expr_real_norm})

                # get loss
                d_loss, g_loss, p_loss, s_loss, e_loss, id_loss, counter = self.sess.run(
                                                        [self.d_loss, self.g_loss,self.p_loss,self.s_loss,self.e_loss,
                                                         self.id_loss, self.global_step],
                                                        feed_dict={self.images: batch_images,
                                                                   self.z: batch_z,
                                                                   self.shape_real: shape_real_norm,
                                                                   self.pose_real: pose_real_norm,
                                                                   self.expr_real: expr_real_norm})

                # Update learning rate
                if epoch % self.cfg.lr_drop_period == 0 and idx == (batch_idxs-1):

                    _, _, self.g_lr_, self.d_lr_, = self.sess.run([self.g_lr, self.d_lr, self.g_lr_update, self.d_lr_update],
                                                                  feed_dict={self.images: batch_images,
                                                                             self.z: batch_z,
                                                                             self.shape_real: shape_real_norm,
                                                                             self.pose_real: pose_real_norm,
                                                                             self.expr_real: expr_real_norm})
                    print('Update learning rate once....')
                # add to summary
                if counter % self.cfg.add_summary_period == 0:
                    [merged_] = self.sess.run([merged],feed_dict={self.images: batch_images,
                                                                  self.z: batch_z,
                                                                  self.shape_real: shape_real_norm,
                                                                  self.pose_real: pose_real_norm,
                                                                  self.expr_real: expr_real_norm})
                    self.writer.add_summary(merged_, counter)
                    print('add sunmmary once....')

                endtime = datetime.datetime.now()
                timediff = (endtime - starttime).total_seconds()
                print(
                    "Epoch: [%2d/%2d] [%5d/%5d] time:%3.2f, d_loss:%.4f, g_loss:%.4f, p_loss:%.5f, s_loss:%.5f, e_loss:%.5f, id_loss:%.5f, k_t:%.6f, Mglobal:%.6f, g_lr:%.6f, d_lr:%.6f" \
                    % (epoch, self.cfg.epoch, idx, batch_idxs, timediff, d_loss, g_loss,p_loss,s_loss,e_loss,
                       id_loss, k_t, Mglobal, self.g_lr_, self.d_lr_))
                try:
                    if epoch % self.cfg.saveimage_period == 0 and idx % self.cfg.saveimage_idx == 0:
                        samples = self.sess.run([self.sampler], feed_dict={self.images: batch_images_G,
                                                                           self.z: batch_z_G,
                                                                           self.shape_real: shape_real_norm_G,
                                                                           self.pose_real: pose_real_norm_G,
                                                                           self.expr_real: expr_real_norm_G})
                        m4_image_save_cv(samples[0], '{}/train_{}_{}.jpg'.format(self.cfg.sampel_save_dir, epoch, counter))
                        print('save train_{}_{}.jpg image.'.format(epoch, counter))
                except:
                    print('one picture save error....')

                try:
                    if epoch % self.cfg.savemodel_period == 0 and idx % 10000 == 0:
                        self.save(self.cfg.checkpoint_dir, epoch, self.cfg.dataset_name)
                        print('one param model saved....')
                except:
                    print('one model save error....')


        # except:
        #     print('Mission complete!')
        '''

    def test(self):
        try:
            self.saver = tf.train.Saver()
        except:
            print('one model save error....')
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # load 3DMM model
        self.load_expr_shape_pose_param()

        # load all train param
        could_load, counter = self.load(self.cfg.checkpoint_dir, self.cfg.dataset_name)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        one_element, dataset_size = data_loader(self.cfg.datalabel_dir, self.cfg.datalabel_name, self.cfg.dataset_dir,
                                                self.cfg.dataset_name, self.cfg.batch_size, self.cfg.epoch)
        batch_idxs = dataset_size // (self.cfg.batch_size)
        batch_images_G, batch_labels_G = self.sess.run(one_element)
        batch_z_G = np.random.uniform(-1, 1, [self.cfg.batch_size, self.cfg.z_dim]).astype(np.float32)
        (shape_real_norm_G, expr_real_norm_G, pose_real_norm_G) = self.new_sess.run(
            [self.shape_real_norm, self.expr_real_norm, self.pose_real_norm],
            feed_dict={self.images_new_graph: batch_images_G})
        counter = 0

        for idx in range(1, batch_idxs + 1):
            counter += 1

            batch_images, batch_labels = self.sess.run(one_element)
            batch_z = np.random.uniform(-1, 1, [self.cfg.batch_size * self.cfg.num_gpus, self.cfg.z_dim]).astype(
                np.float32)
            if batch_images.shape[0] < self.cfg.batch_size * self.cfg.num_gpus:
                for add_idx in range(self.cfg.batch_size * self.cfg.num_gpus - batch_images.shape[0]):
                    batch_images = np.append(batch_images, batch_images[0:1], axis=0)

            (shape_real_norm, expr_real_norm, pose_real_norm) = self.new_sess.run(
                [self.shape_real_norm, self.expr_real_norm, self.pose_real_norm],
                feed_dict={self.images_new_graph: batch_images})

            [samples] = self.sess.run([self.sampler], feed_dict={self.images: batch_images,
                                                                 self.z: batch_z,
                                                                 self.shape_real: shape_real_norm,
                                                                 self.pose_real: pose_real_norm,
                                                                 self.expr_real: expr_real_norm})
            m4_image_save_cv(samples, '{}/test_{}.jpg'.format(self.cfg.test_sample_save_dir, counter))
            print('save test_{}.jpg image.'.format(counter))
            m4_image_save_cv(batch_images, '{}/original_{}.jpg'.format(self.cfg.test_sample_save_dir, counter))
            print('save {}/original_{}.jpg'.format(self.cfg.test_sample_save_dir, counter))

    def save(self, checkpoint_dir, step, model_file_name):
        model_name = "GAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, model_file_name)

        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir, model_folder_name):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, model_folder_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            time.sleep(3)
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            time.sleep(3)
            return False, 0
