import tensorflow as tf
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
#tf.add_to_collection('vars', w1)
#tf.add_to_collection('vars', w2)
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#saver.save(sess, '/Users/lipingzhang/Downloads/model/my_test_model')

# **# save method will call export_meta_graph implicitly.
saver.save(sess, '/Users/lipingzhang/Downloads/model/my_test_model', global_step=10)


