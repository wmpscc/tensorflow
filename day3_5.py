#!/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf

sess = tf.InteractiveSession()
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1), name="w1")
w2 = tf.Variable(tf.random_normal([2, 2], stddev=1), name="w2")
sess.run(tf.global_variables_initializer())
print(w1.eval())
print()
print(w2.eval())
print()
tf.assign(w1, w2, validate_shape=False)
print(w1.eval())
print()
print(w2.eval())