import tensorflow as tf
import numpy as np
import os
import datetime
from utils import *
from ops import *
from networks import *
from data_loader_tensorflow_dataset import *
import time
import ExpShapePoseNet as ESP
import scipy
import scipy.io as sio
import utils_3DMM

class my_gan:
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg
        self.new_graph = tf.Graph()

        self.images = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 128, 128, 3], name='real_image')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 10575], name='id')
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.cfg.z_dim], name='noise_z')

        self.shape_real = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 198], name='shape_real')
        self.pose_real = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 6], name='pose_real')
        self.expr_real = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 29], name='expr_real')
        # self.id_real = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 128], name='id_real')

        with self.new_graph.as_default():
            self.images_new_graph = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 128, 128, 3],
                                     name='real_image')


        m4_BE_GAN_model = m4_BE_GAN_network(self.sess, self.cfg, self.new_graph)
        m4_BE_GAN_model.build_model(self.images, self.labels, self.z, self.images_new_graph, self.shape_real, self.pose_real,
                                    self.expr_real)
        self.g_optim = m4_BE_GAN_model.g_optim
        self.d_optim = m4_BE_GAN_model.d_optim
        self.g_loss = m4_BE_GAN_model.g_loss
        self.d_loss = m4_BE_GAN_model.d_loss
        self.p_loss = m4_BE_GAN_model.pose_loss
        self.s_loss = m4_BE_GAN_model.shape_loss
        self.e_loss = m4_BE_GAN_model.expr_loss
        self.id_loss = m4_BE_GAN_model.id_loss

        self.global_step = m4_BE_GAN_model.global_step
        self.sampler = m4_BE_GAN_model.G
        self.k_update = m4_BE_GAN_model.k_update
        self.k_t = m4_BE_GAN_model.k_t
        self.Mglobal = m4_BE_GAN_model.measure
        self.d_lr_update = m4_BE_GAN_model.d_lr_update
        self.g_lr_update = m4_BE_GAN_model.g_lr_update
        self.d_lr = m4_BE_GAN_model.d_lr
        self.g_lr = m4_BE_GAN_model.g_lr
        self.g_lr_ = self.cfg.g_lr
        self.d_lr_ = self.cfg.d_lr

        self.shape_real_norm = m4_BE_GAN_model.shape_real_norm
        self.expr_real_norm = m4_BE_GAN_model.expr_real_norm
        self.pose_real_norm = m4_BE_GAN_model.pose_real_norm
        self.new_sess = m4_BE_GAN_model.new_sess

        # self.shape_norm = m4_BE_GAN_model.shape_norm
        # self.expr_norm = m4_BE_GAN_model.expr_norm
        # self.pose_norm = m4_BE_GAN_model.pose_norm

    def train(self):
        try:
            self.saver = tf.train.Saver()
        except:
            print('one model save error....')
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.writer = tf.summary.FileWriter('{}/{}'.format(self.cfg.log_dir,
                                                           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))),
                                                           self.sess.graph)
        merged = tf.summary.merge_all()

        # load pre_model
        could_load, counter = self.load(self.cfg.BE_GAN_model_dir, self.cfg.BE_GAN_model_name)

        # load face_model
        t_vars = tf.trainable_variables()
        face_vars = [var for var in t_vars if 'facenet' in var.name]
        face_model_saver = tf.train.Saver(face_vars)
        self.load_face_model(face_model_saver,self.cfg.face_model_dir, self.cfg.face_model_name)
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

        m4_image_save_cv(batch_images_G, '{}/x_fixed.jpg'.format(self.cfg.sampel_save_dir))
        print('save x_fixed.jpg.')
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
                    batch_images = np.append(batch_images,batch_images[0:1],axis=0)

            (shape_real_norm, expr_real_norm, pose_real_norm) = self.new_sess.run(
                                                                [self.shape_real_norm, self.expr_real_norm, self.pose_real_norm],
                                                                feed_dict={self.images_new_graph: batch_images})


            [samples] = self.sess.run([self.sampler], feed_dict={self.images: batch_images,
                                                               self.z: batch_z,
                                                               self.shape_real: shape_real_norm,
                                                               self.pose_real: pose_real_norm,
                                                               self.expr_real: expr_real_norm})
            m4_image_save_cv(samples, '{}/test_{}.jpg'.format(self.cfg.test_sample_save_dir,counter))
            print('save test_{}.jpg image.'.format(counter))
            m4_image_save_cv(batch_images, '{}/original_{}.jpg'.format(self.cfg.test_sample_save_dir, counter))
            print('save {}/original_{}.jpg'.format(self.cfg.test_sample_save_dir, counter))



    def ESP_test(self):
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

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
            saver_pose.restore(self.sess, load_path)
            print('Load ' + self.cfg.fpn_new_model_ckpt_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.fpn_new_model_ckpt_file_path + ' failed....')

        # load 3dmm shape and texture model from Tran et al.' CVPR2017
        try:
            load_path = self.cfg.Shape_Model_file_path
            saver_ini_shape_net.restore(self.sess, load_path)
            print('Load ' + self.cfg.Shape_Model_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.Shape_Model_file_path + ' failed....')

        # load our expression net model
        try:
            load_path = self.cfg.Expression_Model_file_path
            saver_ini_expr_net.restore(self.sess, load_path)
            print('Load ' + self.cfg.Expression_Model_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.Expression_Model_file_path + ' failed....')

        could_load, counter = self.load(self.cfg.checkpoint_dir, self.cfg.dataset_name)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        names = np.loadtxt(os.path.join(self.cfg.datalabel_dir, self.cfg.datalabel_name), dtype=np.str)
        dataset_size = names.shape[0]
        names, labels = m4_get_file_label_name(os.path.join(self.cfg.datalabel_dir, self.cfg.datalabel_name),
                                               os.path.join(self.cfg.dataset_dir, self.cfg.dataset_name))
        filenames = tf.constant(names)
        filelabels = tf.constant(labels)
        try:
            dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
        except:
            dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, filelabels))

        dataset = dataset.map(m4_parse_function)
        dataset = dataset.shuffle(buffer_size=10000).batch(self.cfg.batch_size * self.cfg.num_gpus).repeat(
            self.cfg.epoch)
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        batch_idxs = dataset_size // (self.cfg.batch_size * self.cfg.num_gpus)
        batch_images_G, batch_labels_G = self.sess.run(one_element)
        batch_z_G = np.random.uniform(-1, 1, [self.cfg.batch_size * self.cfg.num_gpus, self.cfg.z_dim]).astype(
            np.float32)
        m4_image_save_cv(batch_images_G,
                         '{}/x_fixed.jpg'.format(self.cfg.mesh_folder))
        print('save x_fixed.jpg.')

        if not os.path.exists(self.cfg.mesh_folder):
            os.makedirs(self.cfg.mesh_folder)

        # x = tf.placeholder(tf.float32, [self.cfg.batch_size * self.cfg.num_gpus, 256, 256, 3])



        print('> Start to estimate Expression, Shape, and Pose!')

        image = cv2.imread('/home/yang/My_Job/study/Gan_Network/BE_GAN_MutiGPU_With_ID/subject1_a.jpg', 1)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        image_size_h, image_size_w, nc = image.shape
        image = image / 127.5 - 1.0

        image1 = cv2.imread('/home/yang/My_Job/study/Gan_Network/BE_GAN_MutiGPU_With_ID/subject15_a.jpg', 1)  # BGR
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = cv2.resize(image1,(256,256),interpolation=cv2.INTER_CUBIC)
        image_size_h, image_size_w, nc = image1.shape
        image1 = image1 / 127.5 - 1.0

        image_list = []
        image_list.append(image)
        image_list.append(image1)

        image_np = np.asarray(image_list)
        image_np = np.reshape(image_np, [self.cfg.batch_size * 2, image_size_h, image_size_w, 3])

        (Shape_Texture, Expr, Pose) = self.sess.run([self.fc1ls, self.fc1le, self.pose_model], feed_dict={self.images: image_np})
        print(Shape_Texture)
        # -------------------------------make .ply file---------------------------------
        ## Modifed Basel Face Model
        BFM_path = self.cfg.BaselFaceModel_mod_file_path
        model = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
        model = model["BFM"]
        faces = model.faces - 1
        print('> Loaded the Basel Face Model to write the 3D output!')

        for i in range(self.cfg.batch_size * self.cfg.num_gpus):
            outFile = self.cfg.mesh_folder + '/' + 'haha' + '_' + str(i)

            Pose[i] = np.reshape(Pose[i], [-1])
            Shape_Texture[i] = np.reshape(Shape_Texture[i], [-1])
            Shape = Shape_Texture[i][0:99]
            Shape = np.reshape(Shape, [-1])
            Expr[i] = np.reshape(Expr[i], [-1])

            #########################################
            ### Save 3D shape information (.ply file)

            # Shape + Expression + Pose
            SEP, TEP = utils_3DMM.projectBackBFM_withEP(model, Shape_Texture[i], Expr[i], Pose[i])
            utils_3DMM.write_ply_textureless(outFile + '_Shape_Expr_Pose.ply', SEP, faces)


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
    def load_face_model(self, saver, checkpoint_dir, model_folder_name):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, model_folder_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            time.sleep(3)
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            time.sleep(3)
            return False, 0

    def load_expr_shape_pose_param(self):
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
            saver_pose.restore(self.sess, load_path)
            print('Load ' + self.cfg.fpn_new_model_ckpt_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.fpn_new_model_ckpt_file_path + ' failed....')

        # load 3dmm shape and texture model from Tran et al.' CVPR2017
        try:
            load_path = self.cfg.Shape_Model_file_path
            saver_ini_shape_net.restore(self.sess, load_path)
            print('Load ' + self.cfg.Shape_Model_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.Shape_Model_file_path + ' failed....')

        # load our expression net model
        try:
            load_path = self.cfg.Expression_Model_file_path
            saver_ini_expr_net.restore(self.sess, load_path)
            print('Load ' + self.cfg.Expression_Model_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.Expression_Model_file_path + ' failed....')
            time.sleep(3)
