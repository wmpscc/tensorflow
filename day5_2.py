#!/usr/bin/env python
#_*_coding:utf-8_*_
from numpy.random import RandomState
import tensorflow as tf
weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])

with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))  # L1正则化
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))  # L2正则化
    # print(sess.run(tf.contrib.layers.apply_regularization(.5)(weights)))
    # print(sess.run(tf.contrib.layers.sum_regularizer(.5)(weights)))