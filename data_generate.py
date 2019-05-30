#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import config as cfg
import os
import cv2 as cv
import pickle


def rotateImage1(img,degree):
    w, h, depth = img.shape
    img_change = cv.getRotationMatrix2D((w / 2, h / 2), degree, 1)
    res = cv.warpAffine(img, img_change, (w, h))
    return res


class data_manage():
    def __init__(self, net_type):
        self.cursor = 0
        self.epoch = 0
        self.val_cursor = 0
        self.database, self.character_name = self.get_database(net_type + ".pkl")
        # 用来将label和图片中文类别进行对应
        self.character_to_label = dict(list(zip(list(range(len(self.character_name))), self.character_name)))
        print(self.character_to_label)
        np.random.shuffle(self.database)


    def get_database(self, net_type):
        pkl_file = os.path.join(cfg.cache_path, net_type)
        train_path = cfg.train_path
        all_dir = os.listdir(train_path)
        if os.path.isfile(pkl_file) and not cfg.cache_rebuild:
            print('Loading gt_database from: ' + pkl_file)
            with open(pkl_file, "rb") as f:
                database = pickle.load(f)
            return database, all_dir

        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)

        print("reloading gt database from Tiny-Mind data")
        database = []
        all_label = []
        for label, sub_dir in enumerate(all_dir):
            lists, labels = self.data(label, train_path, sub_dir, "trainval")
            database.extend(lists)
            all_label.extend(labels)
        print("reload succeeded")
        with open(pkl_file, "wb") as f:
            pickle.dump(database, f)

        return database, all_dir

    def data(self, sub_label, up_path, dir_name, net_type): #
        if net_type == "trainval":
            sub_trainval_list = []
            sub_labels = []
            sub_path = os.path.join(up_path, dir_name)
            image_name_list = os.listdir(sub_path)
            # print(sub_path)
            for image_name in image_name_list:
                image_path = os.path.join(sub_path, image_name)
                img = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
                image_size = img.shape
                image_label = sub_label
                sub_dict = {"image_path": image_path, "image_name": image_name,
                            "image_size": image_size, "image_label": image_label}
                sub_trainval_list.append(sub_dict)
                sub_labels.append(sub_label)

            return sub_trainval_list, sub_labels

    def data_zip(self):
        origin_image_array = np.empty((cfg.batch_size, cfg.image_height, cfg.image_width, cfg.image_channels),
                                      dtype=np.float32)
        origin_label = np.zeros((cfg.batch_size,), dtype=np.int32)
        image_begin_index = self.cursor * cfg.batch_size
        # using_image_index = []
        # for i in range(cfg.batch_size):
        #     using_image_index.append(image_begin_index + i)
        # print(using_image_index)
        # cv.waitKey()
        for index in range(cfg.batch_size):
            image = cv.imdecode(np.fromfile(self.database[image_begin_index + index]["image_path"], dtype=np.uint8), 1)
            image = cv.resize(image, (cfg.image_height, cfg.image_width))
            image = rotateImage1(image,np.random.randint(-10,10))
            # cv.imshow('image',image)
            # cv.imshow('image1', image1)
            # cv.waitKey()
            if cfg.image_channels == 1:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                image = np.reshape(image, [cfg.image_height, cfg.image_width, 1])
            origin_image_array[int(index), :, :, :] = image
            origin_label[int(index)] = self.database[image_begin_index + index]["image_label"]
            # cv.imshow(str(index), image)
            # print(self.character_to_label[origin_label[int(index)]])
            # cv.waitKey(5000)
        # cv.waitKey(5000)
        # cv.destroyAllWindows()
        self.cursor += 1
        if self.cursor >= int(len(self.database) / cfg.batch_size):
            np.random.shuffle(self.database)
            self.cursor = 0
            self.epoch += 1

        return origin_image_array, origin_label

    # def val_data_zip(self):
    #     val_origin_image_array = np.empty((cfg.val_batch_size, cfg.image_height, cfg.image_width,
    #                                        cfg.image_channels), dtype=np.float32)
    #     val_origin_label = np.zeros((cfg.val_batch_size,), dtype=np.int32)
    #     val_image_begin_index = self.val_cursor * cfg.val_batch_size
    #     val_using_image_index = []
    #     for i in range(cfg.val_batch_size):
    #         val_using_image_index.append(val_image_begin_index + i)
    #     for index in range(cfg.val_batch_size):
    #         image = cv.imdecode(
    #             np.fromfile(self._val_database[val_image_begin_index + index]["image_path"], dtype=np.uint8), 1)
    #         image = cv.resize(image, (cfg.image_height, cfg.image_width))
    #         if cfg.image_channels == 1:
    #             image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #             image = np.reshape(image, [cfg.image_height, cfg.image_width, 1])
    #         val_origin_image_array[int(index), :, :, :] = image
    #         val_origin_label[int(index)] = self._val_database[val_image_begin_index + index]["image_label"]
    #
    #     self.val_cursor += 1
    #     if self.val_cursor >= int(len(self._val_database) / cfg.val_batch_size):
    #         np.random.shuffle(self._val_database)
    #         self.val_cursor = 0
    #
    #     return val_origin_image_array, val_origin_label


if __name__ == "__main__":
    data = data_manage("trainval")
