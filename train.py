import config as cfg
import tensorflow as tf
import resnet
import data_generate
import numpy as np
import os


def class_loss(labels, cls_scores):
    labels = tf.reshape(labels, [-1]) #选取的roi的target label 前景是类别标签 背景是0
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_scores, labels=labels))
    return cross_entropy    


def train(img_data, is_training):
    # 制作placeholder
    #initial_learning_rate = tf.Variable(cfg.lr[0],trainable=False)
    input_img = tf.placeholder(tf.float32,[None, cfg.image_height, cfg.image_height, 3])
    img_label = tf.placeholder(tf.int32, [None])

    ckpt_filename = os.path.join('output_2', 'output.model')
    summary_dir = os.path.join('summary')

    # 得到损失和准确率
    cls_score = resnet.resnet_base(input_img, scope_name=cfg.NET_NAME, is_training=is_training)
    y_hat = tf.argmax(cls_score, 1)
    correct_pred = tf.equal(tf.cast(img_label, tf.int64), y_hat)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss = class_loss(img_label, cls_score)

    # 建立tensorboard的变量
    tf.summary.scalar("total_loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    global_step = tf.train.create_global_step()
    learning_rate = tf.train.exponential_decay(cfg.lr[0], global_step = global_step, decay_steps=50000,
                                                decay_rate=0.1, staircase=False, name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate, cfg.beta1, cfg.beta2)

    # 是否使用BN
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([tf.group(*update_op)]):
        train_op = optimizer.minimize(loss, global_step=global_step)

    # tensorboard显示
    writer_train = tf.summary.FileWriter("./summary/train")
    writer_eval = tf.summary.FileWriter("./summary/val")

    init = tf.global_variables_initializer()

    # 内存生长法占据GPU
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    with tf.Session(config=conf) as sess:
        sess.run(init)
        variables = tf.global_variables()
        saver = tf.train.Saver(variables, max_to_keep=25)
        #saver.restore(sess, "output_2/output.model-32000")
        merged = tf.summary.merge_all()
        for i in range(cfg.num_iters+1):
            #learning_rate = tf.train.exponential_decay(learning_rate, global_step=i, decay_steps=50000,
            #                                           decay_rate=0.1, staircase=False)
            a, b = img_data.data_zip()
            b = np.reshape(b, -1)
            feed_dict = {input_img: a, img_label: b}
            train_accuracy, total_loss, _, lr = sess.run([accuracy, loss, train_op, learning_rate], feed_dict=feed_dict)
            if i % 20 ==0:
                # train
                train_summary = sess.run(merged, feed_dict=feed_dict)
                writer_train.add_summary(train_summary, i)
                print('step %d , train loss: %f , train accuracy: %f, learning_rate: %f' % (i, total_loss, train_accuracy, lr))

                if i % cfg.save_stp == 0:
                    saver.save(sess, ckpt_filename, global_step = i)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data = data_generate.data_manage("trainval")
    train(data, is_training=True)
