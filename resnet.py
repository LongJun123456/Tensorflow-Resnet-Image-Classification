# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import config as cfgs
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils  
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
def resnet_arg_scope(
        is_training, weight_decay=cfgs.WEIGHT_DECAY, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': is_training, 'decay': 0.95,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': is_training,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    } #在训练中不使用这层

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay), #L2正则化
            #weights_initializer=slim.variance_scaling_initializer(),
            weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

def resnet_base(img_batch, scope_name, is_training):
    '''
    this code is derived from light-head rcnn.
    https://github.com/zengarden/light_head_rcnn

    It is convenient to freeze blocks. So we adapt this mode.
    '''
    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23 #101第3个block是23
    elif scope_name == 'resnet_v1_152':
        middle_num_units = 36
    else:
        raise NotImplementedError('We only support resnet_v1_50 、resnet_v1_101 、resnet152. Check your network name....yjr')
                
    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=1),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=2),
              resnet_v1_block('block4', base_depth=512, num_units=3, stride=2)]
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)): #resnet_arg_scope配置参数
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1') #RESNET第一个卷积层， 7*7*64， stride=2
            
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]]) #padding 0 ?? 类似与后面的samepadding?
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1') #3*3最大池化              
    #not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True] #不冻结的Blocks层
    #net = tf.Print(net, [tf.shape(net)], summarize=10, message='net')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                blocks[0:1], #传入的是一个resnet_utils.Block类  一整个Resnet block
                                                global_pool= False,
                                                include_root_block=False,
                                                scope=scope_name) #返回当前构建resnet block层：C2 end_points_C2： collection中已有的特征图 越到后面越多


    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                blocks[1:2],
                                                global_pool = False,
                                                include_root_block=False,
                                                scope=scope_name) #构建第二个block模块


    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C4, end_points_C4 = resnet_v1.resnet_v1(C3,
                                                blocks[2:3],
                                                global_pool = False,
                                                include_root_block=False,
                                                scope=scope_name)


    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5, end_points_C5 = resnet_v1.resnet_v1(C4,
                                                blocks[3:4],
                                                num_classes = cfgs.num_classes,
                                                global_pool = True,
                                                include_root_block=False,
                                                scope=scope_name)
        C5 = tf.reshape(C5, [-1, cfgs.num_classes])
    return C5
    # with tf.variable_scope('cls_layer'):
    #     fc_plat = slim.flatten(C5, scope='flatten')
    #     fc6 = slim.fully_connected(fc_plat, 256, 
    #                                 weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
    #                                 trainable=is_training,
    #                                 weights_regularizer=slim.l2_regularizer(0.0005),
    #                                 activation_fn=None, scope='fc6') #得到class score
    #     cls_score = slim.fully_connected(fc6, cfgs.num_classes, 
    #                                      weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
    #                                      trainable=is_training,
    #                                      weights_regularizer=slim.l2_regularizer(0.0005),
    #                                      activation_fn=None, scope='cls_score') #得到class score
    # return cls_score




























