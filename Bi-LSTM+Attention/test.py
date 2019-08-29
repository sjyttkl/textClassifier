# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     test
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/6/18
   Description :  
==================================================
"""
__author__ = 'songdongdong'


import tensorflow as tf
import numpy as np
arry = np.array([[[10],[3],[2]]])
arryt = tf.convert_to_tensor(arry)

print (np.shape(arry))
print (arryt.eval)
print ("squeeze=>")
with tf.Session() as sess:
    t = tf.squeeze(arryt)
    print(sess.run(t))
    print(sess.run(tf.shape(t)))


