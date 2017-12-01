#!/usr/bin/env python
#_*_coding:utf-8_*_
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def add_layer(inputs,in_size,out_size,activation_function=None):    # size is 神经元个数
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
layer1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediton = add_layer(layer1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediton),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)  # 减小误差

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 图形化窗口
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()   # 持续更新图像
plt.show()

for i in range(10000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediton_value = sess.run(prediton,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediton_value,'r-',lw=5)    # 红色线 宽度为5
        plt.pause(0.1)

