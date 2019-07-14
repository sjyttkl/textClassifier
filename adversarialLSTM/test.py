# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     test
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/6/20
   Description :  
==================================================
"""
__author__ = 'songdongdong'
import tensorflow as tf
a = tf.constant(1.)
b = tf.constant(1.)
c = tf.constant(1.)
g = tf.gradients([5*a + 3*b + 2*c], [a, b, c], stop_gradients=[a, b, c])

sess=tf.Session()
with sess:
    print (sess.run(g))#[5.0, 3.0, 2.0]

#===========================
a = tf.constant(1.)
b = 2*a
c = 3*b
g = tf.gradients([a + b + c], [a, b, c], stop_gradients=[a, b, c])

sess=tf.Session()
with sess:
    print (sess.run(g)) #[1.0, 1.0, 1.0]
#g=a+b+c
#对于a,b,c来说，偏导数g’a、g’b、g’c都是g=1
#==============================
x = tf.constant(1.)
a = 12*x
b = 2*a
c = 3*b
g1 = tf.gradients([a + b + c], [a, b, c])
g2 = tf.gradients([a + b + c], [a, b, c],stop_gradients=[a, b, c])

sess=tf.Session()
with sess:
    print (sess.run(g1))#[9.0, 4.0, 1.0]
    print (sess.run(g2))#[1.0, 1.0, 1.0]

#====================================
x = tf.constant(2.)
a = 12*x
b = 2*a
c = 3*x
g1 = tf.stop_gradient([a,b])
g2 = tf.gradients([a,c],[x])
g3 = tf.gradients([a,b,c],[x])
g4 = tf.gradients([a,b],a)
sess=tf.Session()
with sess:
    print (sess.run(g1))   # [24. 48.]
    print (sess.run(g2))   # [15.0]
    print (sess.run(g3))   # [39.0]
    print (sess.run(g4))   # [3.0]
#+++++++++++++++++++++++++++++++++++++++++++

import tensorflow as tf
import numpy as np

print('******reduce_max******')
a=np.array([[[1, 2],
            [5, 3],
            [2, 9]],

            [[1, 2],
            [5, 3],
            [2,10]]
            ])

b = tf.Variable(a)#(2,3,2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
    print('**************************************************************************')
    # # 对于二维矩阵，axis=0轴可以理解为行增长方向（向下）,axis=1轴可以理解为列增长方向(向右）
    print(sess.run(tf.reduce_max(b, axis=1, keepdims=False)))  # keepdims=False,axis=1被消减,进去一个维度，
    print('************')
    print(sess.run(tf.reduce_max(b, axis=1, keepdims=True))," shape  ",sess.run(tf.shape(tf.reduce_max(b, axis=1, keepdims=True))))
    print('************')
    print(sess.run(tf.reduce_max(b, axis=0, keepdims=True)))
    print('************')
    print(sess.run(tf.reduce_max(b, 2, keepdims=True)))
    print('************')
    b = sess.run(tf.reduce_max(b, (1, 2), keepdims=True))
    print(b)
    print(tf.shape(b))


