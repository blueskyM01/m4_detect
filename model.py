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

        self.m4_DataReader = m4_ReadData(is_train=self.cfg.is_train, dataset_dir=self.cfg.dataset_dir,
                                    dataset_name=self.cfg.dataset_name,
                                    label_dir=self.cfg.datalabel_dir, label_name=self.cfg.datalabel_name,
                                    anchors_path=self.cfg.achorfile_path, class_path=self.cfg.class_path,
                                    num_classes=self.cfg.num_classes,
                                    max_boxes=self.cfg.max_boxes, input_shape=self.cfg.input_shape,
                                    batch_size=self.cfg.batch_size,
                                    epoch=self.cfg.epoch, buffer_size=500)
        self.class_name = self.m4_DataReader.class_name
        self.anchors = self.m4_DataReader.anchors

        self.images = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 416, 416, 3], name='input_image')
        self.bbox_13 = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 13, 13, 3, 85], name='bbox_13')
        self.bbox_26 = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 26, 26, 3, 85], name='bbox_26')
        self.bbox_52 = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size, 52, 52, 3, 85], name='bbox_52')
        self.input_image_shape = tf.placeholder(dtype=tf.int32, shape=(2,))
        m4_yolo_model = m4_yolo_network(self.sess, self.cfg)
        m4_yolo_model.build_model(self.images, self.bbox_13, self.bbox_26, self.bbox_52, self.input_image_shape)
        self.global_step = m4_yolo_model.global_step
        if self.cfg.is_train:
            self.optim = m4_yolo_model.optim
            self.loss = m4_yolo_model.loss
            self.lr = m4_yolo_model.lr_
        else:
            # self.boxes = m4_yolo_model.boxes
            # self.scores = m4_yolo_model.scores
            # self.classes = m4_yolo_model.class_names
            self.box_confidence_list = m4_yolo_model.box_confidence_list
            self.box_class_probs_list = m4_yolo_model.box_class_probs_list
            self.pred_xy_list = m4_yolo_model.pred_xy_list
            self.pred_wh_list = m4_yolo_model.pred_wh_list


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

        # load all train param
        # could_load, counter = self.load(self.cfg.checkpoint_dir, self.cfg.dataset_name)
        # if could_load:
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")


        one_element, dataset_size = self.m4_DataReader.data_loader()

        # [boxes_sum] = self.sess.run([self.boxes_sum], feed_dict={self.images: image_decoded,
        #                                                                    self.bbox_13: bbox_true_13,
        #                                                                    self.bbox_26: bbox_true_26,
        #                                                                    self.bbox_52: bbox_true_52})
        #
        # print(boxes_sum[0])
        # self.writer.add_summary(merged_, counter)
        # self.writer.flush()
        # print('add sunmmary once....')
        #
        #
        # self.m4_PlotOnOriginalImageWithLabeledBox(image_decoded, bbox_true_13, bbox_true_26, bbox_true_52)


        batch_idxs = dataset_size // (self.cfg.batch_size)


        # try:
        for epoch in range(1,self.cfg.epoch+1):
            for idx in range(1, batch_idxs + 1):
                starttime = datetime.datetime.now()
                image_decoded, boxes_data_available, bbox_true_13, bbox_true_26, bbox_true_52 = self.sess.run(one_element)

                # 补全batch
                if image_decoded.shape[0] < self.cfg.batch_size:
                    for add_idx in range(self.cfg.batch_size - image_decoded.shape[0]):
                        image_decoded = np.append(image_decoded,image_decoded[0:1],axis=0)
                        boxes_data_available = np.append(boxes_data_available,boxes_data_available[0:1],axis=0)
                        bbox_true_13 = np.append(bbox_true_13,bbox_true_13[0:1],axis=0)
                        bbox_true_26 = np.append(bbox_true_26, bbox_true_26[0:1], axis=0)
                        bbox_true_52 = np.append(bbox_true_52, bbox_true_52[0:1], axis=0)


                # get loss
                loss, counter, _, lr = self.sess.run([self.loss, self.global_step, self.optim,self.lr],
                                                        feed_dict={self.images: image_decoded,
                                                                   self.bbox_13: bbox_true_13,
                                                                   self.bbox_26: bbox_true_26,
                                                                   self.bbox_52: bbox_true_52})


                # add to summary
                if counter % self.cfg.add_summary_period == 0:
                    [merged_] = self.sess.run([merged], feed_dict={self.images: image_decoded,
                                                                   self.bbox_13: bbox_true_13,
                                                                   self.bbox_26: bbox_true_26,
                                                                   self.bbox_52: bbox_true_52})
                    self.writer.add_summary(merged_, counter)
                    print('add sunmmary once....')

                endtime = datetime.datetime.now()
                timediff = (endtime - starttime).total_seconds()
                print("Epoch: [%2d/%2d] [%5d/%5d] time:%3.2f, loss:%.6f, lr:%.6f" % (epoch, self.cfg.epoch, idx, batch_idxs, timediff, loss, lr))

                try:
                    if epoch % self.cfg.savemodel_period == 0 and idx == 1:
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

        # load all train param
        could_load, counter = self.load(self.cfg.checkpoint_dir, self.cfg.dataset_name)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        one_element, dataset_size = self.m4_DataReader.data_loader()
        image_decoded, boxes_data_available, bbox_true_13, bbox_true_26, bbox_true_52 = self.sess.run(one_element)

        box_confidence, box_class_probs, pred_xy,  pred_wh = self.sess.run([self.box_confidence_list[0], self.box_class_probs_list[0],
                                                       self.pred_xy_list[0], self.pred_wh_list[0]],
                                             feed_dict={self.images: image_decoded})

        # anchors_1 = np.reshape(self.anchors[6, 7, 8], [1, 1, 3, 2])

        font = cv2.FONT_HERSHEY_SIMPLEX
        for img, confidence, class_probs, xy, wh in zip(image_decoded, box_confidence, box_class_probs, pred_xy, pred_wh):
            counter += 1
            img = (img * 255.0).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # cv2默认为bgr顺序
            for row in range(confidence.shape[0]):
                for col in range(confidence.shape[1]):
                    for anchor in range(3):
                        if confidence[row][col][anchor][0] > 0.9:
                            center_x = (xy[row][col][anchor][0]) * 416.0
                            center_y = (xy[row][col][anchor][1]) * 416.0

                            x_min = center_x - wh[row][col][anchor][0] * 416. / 2.0
                            y_min = center_y - wh[row][col][anchor][1] * 416. / 2.0
                            x_max = center_x + wh[row][col][anchor][0] * 416. / 2.0
                            y_max = center_y + wh[row][col][anchor][1] * 416. / 2.0
                            if x_min < 0:
                                x_min =0.
                            if x_min > 415.:
                                x_min = 415
                            if y_min < 0:
                                y_min =0.
                            if y_min > 415.:
                                y_min = 415
                            if x_max < 0:
                                x_max =0.
                            if x_max > 415.:
                                x_max = 415
                            if y_max < 0:
                                y_max =0.
                            if y_max > 415.:
                                y_max = 415

                            tl = (int(x_min), int(y_min))
                            br = (int(x_max), int(y_max))
                            print(tl)
                            print(br)
                            # cat = np.argmax(label_13[y_idx][x_idx][anchor_idx][5:])
                            # cv2.putText(img, self.class_name[cat], tl, font, 0.5, (255, 0, 0), 1)
                            cv2.rectangle(img, tl, br, (0, 0, 255), 2)
            cv2.imshow(str(counter), img)
        cv2.waitKey(0)





        print(box_confidence[0].shape)
        print(box_class_probs[0].shape)
        print(pred_xy[0].shape)
        print(pred_wh[0].shape)

        # self.m4_PlotOnOriginalImageWithLabeledBox(image_decoded, bbox_true_13, bbox_true_26, bbox_true_52)




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

    def m4_PlotOnOriginalImageWithLabeledBox(self, image_decoded, bbox_true_13, bbox_true_26, bbox_true_52):
        counter = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        for img, label_13, label_26, label_52 in zip(image_decoded, bbox_true_13, bbox_true_26, bbox_true_52):
            counter += 1
            img = (img * 255.0).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # cv2默认为bgr顺序
            for y_idx in range(label_13.shape[0]):
                for x_idx in range(label_13.shape[1]):
                    for anchor_idx in range(label_13.shape[2]):
                        y_center = label_13[y_idx][x_idx][anchor_idx][1] * 416.0
                        x_center = label_13[y_idx][x_idx][anchor_idx][0] * 416.0
                        y_min = y_center - label_13[y_idx][x_idx][anchor_idx][3] * 416.0 / 2.0
                        x_min = x_center - label_13[y_idx][x_idx][anchor_idx][2] * 416.0 / 2.0
                        y_max = y_center + label_13[y_idx][x_idx][anchor_idx][3] * 416.0 / 2.0
                        x_max = x_center + label_13[y_idx][x_idx][anchor_idx][2] * 416.0 / 2.0
                        if label_13[y_idx][x_idx][anchor_idx][4] > 0:
                            tl = (int(x_min), int(y_min))
                            br = (int(x_max), int(y_max))
                            cat = np.argmax(label_13[y_idx][x_idx][anchor_idx][5:])
                            cv2.putText(img, self.class_name[cat], tl, font, 0.5, (255, 0, 0), 1)
                            cv2.rectangle(img, tl, br, (0, 0, 255), 2)

            for y_idx in range(label_26.shape[0]):
                for x_idx in range(label_26.shape[1]):
                    for anchor_idx in range(label_26.shape[2]):
                        y_center = label_26[y_idx][x_idx][anchor_idx][1] * 416.0
                        x_center = label_26[y_idx][x_idx][anchor_idx][0] * 416.0
                        y_min = y_center - label_26[y_idx][x_idx][anchor_idx][3] * 416.0 / 2.0
                        x_min = x_center - label_26[y_idx][x_idx][anchor_idx][2] * 416.0 / 2.0
                        y_max = y_center + label_26[y_idx][x_idx][anchor_idx][3] * 416.0 / 2.0
                        x_max = x_center + label_26[y_idx][x_idx][anchor_idx][2] * 416.0 / 2.0
                        if label_26[y_idx][x_idx][anchor_idx][4] > 0:
                            tl = (int(x_min), int(y_min))
                            br = (int(x_max), int(y_max))
                            cat = np.argmax(label_26[y_idx][x_idx][anchor_idx][5:])
                            cv2.putText(img, self.class_name[cat], tl, font, 0.5, (255, 0, 0), 1)
                            cv2.rectangle(img, tl, br, (0, 0, 255), 2)

            for y_idx in range(label_52.shape[0]):
                for x_idx in range(label_52.shape[1]):
                    for anchor_idx in range(label_52.shape[2]):
                        y_center = label_52[y_idx][x_idx][anchor_idx][1] * 416.0
                        x_center = label_52[y_idx][x_idx][anchor_idx][0] * 416.0
                        y_min = y_center - label_52[y_idx][x_idx][anchor_idx][3] * 416.0 / 2.0
                        x_min = x_center - label_52[y_idx][x_idx][anchor_idx][2] * 416.0 / 2.0
                        y_max = y_center + label_52[y_idx][x_idx][anchor_idx][3] * 416.0 / 2.0
                        x_max = x_center + label_52[y_idx][x_idx][anchor_idx][2] * 416.0 / 2.0
                        if label_52[y_idx][x_idx][anchor_idx][4] > 0:
                            tl = (int(x_min), int(y_min))
                            br = (int(x_max), int(y_max))
                            cat = np.argmax(label_52[y_idx][x_idx][anchor_idx][5:])
                            cv2.putText(img, self.class_name[cat], tl, font, 0.5, (255, 0, 0), 1)
                            cv2.rectangle(img, tl, br, (0, 0, 255), 2)

            cv2.imshow(str(counter), img)
        cv2.waitKey(0)