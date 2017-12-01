#!/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf
import numpy as np

# creat data
x_data = np.random.rand(1000).astype(np.float32)
y_data = x_data*0.1 + 0.3

### creat tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.6)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

### creat tensorflow structure end ###

sess = tf.Session()
sess.run(init)      # important

for step in range(1000):
    sess.run(train)     # 每run一次执行一次结构
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))
