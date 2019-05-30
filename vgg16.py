import tensorflow as tf
import config as cfg
slim = tf.contrib.slim
def vgg16(input_image, is_training):
    batch_norm_params = {
        'is_training': False, 'decay': cfg.batch_norm_decay,
        'epsilon': cfg.batch_norm_epsilon, 'scale': cfg.batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    } #在训练中不使用这层
    with tf.variable_scope('vgg_16') :
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):                
                net = slim.repeat(input_image, 2, slim.conv2d, 64, [3, 3], trainable=is_training, scope='conv1')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=is_training, scope='conv2')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
                #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                flat = slim.flatten(net, scope='flatten')
                fc6 = slim.fully_connected(flat, 512, scope='fc6') #全连接层
                fc7 = slim.fully_connected(fc6, 512, scope='fc7')#4096*4096
                cls_score = slim.fully_connected(fc7, cfg.num_classes,                 
                                            trainable=is_training,
                                            activation_fn=None, scope='cls_score') #得到class score
    return cls_score