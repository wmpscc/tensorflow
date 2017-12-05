#!/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf

# 创建一个会话
sess = tf.Session()
# 使用这个创建好的会话来得到关心的运算的结果。比如可以调用sess.run(result),
# 来得到张量result的取值
sess.run(...)
# 关闭会话使得本次运行中使用到的资源可以被释放

# 创建一个会话，并通过Python中的上下文管理器来管理这个会话。
with tf.Session() as sess:
    # 使用这创建好的会话来计算关心的结果。
    sess.run(...)
# 不需要再调用“Session.close()”函数来关闭会话，
# 当上下文退出时会话关闭和资源释放也自动完成。

