#!/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf

# tensorflow 中定义一个变量需这样指明它是变量，0是赋的初始值，name=是变量的名字
state = tf.Variable(0,name='counter')
# print(state.name)
one = tf.constant(1)    # constant常量
new_value = tf.add(state,one)
updata = tf.assign(state,new_value)

# init = tf.initialize_all_variables()       # 已被弃用
init = tf.global_variables_initializer()     # must hava if define variable
with tf.Session() as sess:
    sess.run(init)  # important
    for _ in range(3):
        sess.run(updata)
        print(sess.run(state))




