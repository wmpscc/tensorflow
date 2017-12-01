#!/usr/bin/env python
#_*_coding:utf-8_*_

import tensorflow as tf

input1 = tf.placeholder(tf.float32)     # 给定类型
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))

# 用placeholder 意味着要用feed_dict 传值，并且是在用sess.run()运行output时输入值