from build_network import Net
import config as cfg
import tensorflow as tf
import resnet
import data_generate
import numpy as np
import os
import vgg16
import cv2 as cv
import pandas as pd


def class_loss(labels, cls_scores):
    labels = tf.reshape(labels, [-1]) #选取的roi的target label 前景是类别标签 背景是0
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_scores, labels=labels))
    return cross_entropy    


def train(image_data, is_training):
    input_img = tf.placeholder(tf.float32,[None, cfg.image_height, cfg.image_width, 3])
    cls_score = resnet.resnet_base(input_img, scope_name=cfg.NET_NAME, is_training=is_training)
    init = tf.global_variables_initializer()
    output_dir = os.path.join('output_2')
    ckpt_file = tf.train.latest_checkpoint(output_dir)
    variables = tf.global_variables()
    result = []
    saver = tf.train.Saver(variables)
    test_image_list = os.listdir(cfg.test2_path)
    predict_five_all = []
    with tf.Session() as sess:
        variables = tf.global_variables()
        sess.run(init)
        saver.restore(sess, ckpt_file)
        for i in range(int(len(test_image_list))):
            test_image_single_1 = cv.imdecode(np.fromfile(os.path.join(cfg.test2_path, test_image_list[i]),
                                                        dtype=np.uint8), 1)
            
            test_image_single = cv.resize(test_image_single_1, (cfg.image_height, cfg.image_width))
            test_image_single = np.reshape(test_image_single, [1, cfg.image_height, cfg.image_width, 3])
            test_image_single.astype(np.float32)

            feed_dict = {input_img: test_image_single}
            cls_score_1 = sess.run(cls_score, feed_dict=feed_dict)
            result.append(cls_score_1)
            top_5 = np.argsort(cls_score_1, axis=1)
            top_5 = np.reshape(top_5[:, -5::], -1)
            top_5 = top_5[::-1]
            name_label = []
            for j in top_5:
                name_label.extend(image_data.character_name[j])
            # print(name_label)
            # cv.imshow('sprint（image_datahow', test_image_single_1)
            # cv.waitKey(0)
            predict_five = {"name_label": name_label, "test_image_name": test_image_list[i]}
            predict_five_all.append(predict_five)
            if i % 100 == 0:
                print("the processing finished ", i)
        with open("data.txt", "w+") as f:
            f.write(str(result))
        print(len(predict_five_all))
        return predict_five_all


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data = data_generate.data_manage("trainval")
    predict_ = train(data, is_training=False)
    filename = []
    predicts = []
    for f in predict_:
        filename.append(f['test_image_name'])
        str_ = ''.join(f['name_label'])
        predicts.append(str_)
    print(predicts)
    dataframe = pd.DataFrame({'filename': filename, 'label': predicts})
    dataframe.to_csv("res_test.csv", index=False, encoding='utf-8')

    read = pd.read_csv('res_test.csv')
    print(read)
