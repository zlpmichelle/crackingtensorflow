
# coding: utf-8

# In[ ]:


...create graph...
my_train_op = ...

sv = tf.train.Supervisor(logdir="/my/training/directory")
with sv.managed_session() as sess:
  for step in range(100000):
    if sv.should_stop():
      break
    sess.run(my_train_op)


# In[ ]:


...create graph...
 my_train_op = ...
 my_summary_op = tf.summary.merge_all()

 sv = tf.train.Supervisor(logdir="/my/training/directory",
                    summary_op=None) # Do not run the summary service
 with sv.managed_session() as sess:
   for step in range(100000):
     if sv.should_stop():
       break
     if step % 100 == 0:
       _, summ = session.run([my_train_op, my_summary_op])
       sv.summary_computed(sess, summ)
     else:
       session.run(my_train_op)


# In[ ]:


...create graph...
# Create a saver that restores only the pre-trained variables.
pre_train_saver = tf.train.Saver([pre_train_var1, pre_train_var2])

# Define an init function that loads the pretrained checkpoint.
def load_pretrain(sess):
  pre_train_saver.restore(sess, "<path to pre-trained-checkpoint>")

# Pass the init function to the supervisor.
#
# The init function is called _after_ the variables have been initialized
# by running the init_op.
sv = tf.train.Supervisor(logdir="/my/training/directory",
                   init_fn=load_pretrain)
with sv.managed_session() as sess:
  # Here sess was either initialized from the pre-trained-checkpoint or
  # recovered from a checkpoint saved in a previous run of this code.
  ...


# In[ ]:


def my_additional_sumaries(sv, sess):
 ...fetch and write summaries, see below...

...
  sv = tf.train.Supervisor(logdir="/my/training/directory")
  with sv.managed_session() as sess:
    # Call my_additional_sumaries() every 1200s, or 20mn,
    # passing (sv, sess) as arguments.
    sv.loop(1200, my_additional_sumaries, args=(sv, sess))
    ...main training loop...


# In[ ]:


def my_additional_sumaries(sv, sess):
  summaries = sess.run(my_additional_summary_op)
  sv.summary_computed(sess, summaries)


# In[ ]:


# Use a custom Saver and checkpoint every 30 seconds.
  ...create graph...
  my_saver = tf.train.Saver(<only some variables>)
  sv = tf.train.Supervisor(logdir="/my/training/directory",
                     saver=my_saver,
                     save_model_secs=30)
  with sv.managed_session() as sess:
    ...training loop...

