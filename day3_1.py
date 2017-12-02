#!/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    # 在计算图g1中定义变量“v”，并设置初始值0.
    v = tf.get_variable("v",initializer=tf.zeros([1]))

g2 = tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量“v”，并设置初始值为1。
    v = tf.get_variable("v",initializer=tf.ones([1]))

# 在计算图g1中读取变量“v”的取值。
with tf.Session(graph=g1) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    with tf.variable_scope("",reuse=True):
        # 在计算图g1中，变量“v”的取值应该为0，所以下面这行会输出[0.]。
        print(sess.run(tf.get_variable("v")))

# 在计算图g2中读取变量“v”的取值
with tf.Session(graph=g2) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    with tf.variable_scope("",reuse=True):
        # 在计算图g2中，变量“v”的取值应该为1，所以下面这行会输出[1.]。
        print(sess.run(tf.get_variable("v")))


### 使用 tf.Graph.device 函数指定运行计算
g = tf.Graph()
with tf.Session(graph=g) as sess:
    a = tf.constant([1.0,2.0],name="a")
    b = tf.constant([2.0,3.0],name="b")
    with g.device('/cpu:0'):
        result = a + b
        init = tf.global_variables_initializer()
        sess.run(init)
        print(sess.run(result))
        print(result)