#!/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf

# tf.constant是一个计算，这个计算的结果为一个张量，保存在变量a中。
# a = tf.constant([1.0,2.0], name="a")
# b = tf.constant([2.0,3.0], name="b")
# result = tf.add(a,b,name="add")
# print(result)

# a = tf.constant([1, 2], name="a", dtype=tf.float32)
# b = tf.constant([2.0, 3.0], name="b")
# result = tf.add(a, b, name="add")

# 使用张量记录中间的结果
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b

# 直接计算向量的和，这样可读性性会变差
result = tf.constant([1.0, 2.0], name="a") + tf.constant([2.0, 3.0], name="b")

