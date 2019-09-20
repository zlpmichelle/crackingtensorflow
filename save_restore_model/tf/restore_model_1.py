import tensorflow as tf

sess = tf.Session()
new_saver = tf.train.import_meta_graph('/Users/lipingzhang/Downloads/model/my_test_model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('/Users/lipingzhang/Downloads/model/'))
print(sess.run('w1:0'))
print(sess.run('w2:0'))
all_vars = tf.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    print("--" + v_)
    print(sess.run('w1:0'))