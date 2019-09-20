#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
比如，我们需要保存的模型是参数v1和v2，那么只需要使用下列的保存代码save_model.py。
'''

import tensorflow as tf
v1 = tf.Variable(1.1, name="v1")
v2 = tf.Variable(1.2, name="v2")

init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    print v2.eval(sess)
    save_path="/Users/lipingzhang/Downloads/model.ckpt"
    saver.save(sess,save_path)
    print "Model stored...."