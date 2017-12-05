#!/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf

a = tf.constant([1.0],name="a")
b = tf.constant([2.0],name="b")
result = a + b
# sess = tf.Session()
# with sess.as_default():
#     print(result .eval())
# sess = tf.Session()
# print(sess.run(result))
# print(result.eval(session=sess))
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
print(result.eval())
sess1.close()