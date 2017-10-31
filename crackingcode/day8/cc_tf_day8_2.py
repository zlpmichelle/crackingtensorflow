
# coding: utf-8

# In[ ]:


#How to Use TensorFlow Debugger (tfdbg) with tf.contrib.learn

# First, let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)

from tensorflow.python import debug as tf_debug

hooks = [tf_debug.LocalCLIDebugHook()]

#  Create a local CLI debug hook and use it as a monitor when calling fit().
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=1000,
               monitors=hooks)


# In[ ]:


accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target.
                                     hooks=hooks)["accuracy"]


# In[ ]:


#python -m tensorflow.python.debug.examples.debug_tflearn_iris --debug


# In[1]:


# First, let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)

from tensorflow.python import debug as tf_debug

hooks = [tf_debug.LocalCLIDebugHook()]

ex = experiment.Experiment(classifier,
                           train_input_fn=iris_input_fn,
                           eval_input_fn=iris_input_fn,
                           train_steps=FLAGS.train_steps,
                           eval_delay_secs=0,
                           eval_steps=1,
                           train_monitors=hooks,
                           eval_hooks=hooks)

ex.train()
accuracy_score = ex.evaluate()["accuracy"]


# In[ ]:


#python -m tensorflow.python.debug.examples.debug_tflearn_iris  --use_experiment --debug


# In[ ]:


# Let your BUILD target depend on "//tensorflow/python/debug:debug_py
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug

hooks = [tf_debug.DumpingDebugHook("/shared/storage/location/tfdbg_dumps_1")]


# In[ ]:


#python -m tensorflow.python.debug.cli.offline_analyzer \
#    --dump_dir="/shared/storage/location/tfdbg_dumps_1/run_<epoch_timestamp_microsec>_<uuid>"

