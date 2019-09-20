#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
如果，我们要恢复模型，并且把他们导入到变量中，那么首先定义两个参数v3和v4，给他们取名叫v1和v2。注意，这里必须要给v3和v4取名为v1和v2，因为我们保存的模型中给变量取的名字就是v1和v2。那么，模型恢复的代码为restore_model.py
'''
import tensorflow as tf

v3 = tf.Variable(0.0, name="v1")
v4 = tf.Variable(0.0, name="v2")

saver = tf.train.Saver()

with tf.Session() as sess:

    save_path="/Users/lipingzhang/Downloads/model.ckpt"

    saver.restore(sess, save_path)
    print("Model restored.")
    print sess.run(v3)

