
# coding: utf-8

# In[ ]:


# TensorFlow Debugger (tfdbg) Command-Line-Interface Tutorial: MNIST

# python -m tensorflow.python.debug.examples.debug_mnist


# In[1]:


# Let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)

from tensorflow.python import debug as tf_debug

sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)


# In[ ]:


def has_inf_or_nan(datum, tensor):
  return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))


# In[ ]:


#python -m tensorflow.python.debug.examples.debug_mnist --debug


#pt hidden/Relu:0
#run -t 10
#run -f has_inf_or_nan


# In[ ]:


# In python code:
sess.add_tensor_filter('my_filter', my_filter_callable)

# Run at tfdbg run-start prompt:
#tfdbg> run -f my_filter

#tfdbg> pt cross_entropy/Log:0
#tfdbg> /inf
#tfdbg> /(inf|nan)
#tfdbg> ni cross_entropy/Log
#pt softmax/Softmax:0
#tfdbg> /0\.000
#tfdbg> ni -t cross_entropy/Log
#tfdbg> quit

#python -m tensorflow.python.debug.examples.debug_mnist --debug


# In[ ]:


from tensorflow.python.debug import debug_utils

# ... Code where your session and graph are set up...
run_options = tf.RunOptions()
debug_utils.watch_graph(
    run_options,
    session.graph,
    debug_urls=["file:///shared/storage/location/tfdbg_dumps_1"])

# Be sure to use different directories for different run() calls.
session.run(fetches, feed_dict=feeds, options=run_options)

# python -m tensorflow.python.debug.cli.offline_analyzer \
    --dump_dir=/shared/storage/location/tfdbg_dumps_1


# In[ ]:


# Let your BUILD target depend on "//tensorflow/python/debug:debug_py
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python.debubg import debug_utils
sess = tf_debug.DumpingDebugWrapperSession(
    sess, "/shared/storage/location/tfdbg_dumps_1", watch_fn=my_watch_fn)


# In[ ]:


#tfdbg> pt cross_entropy/Log:0[:, 0:10] > /tmp/xent_value_slices.txt

from tensorflow.python import debug as tf_debug

# Then wrap your TensorFlow Session with the local-CLI wrapper.
sess = tf_debug.LocalCLIDebugWrapperSession(sess)


# In[ ]:


# Debugging shape mismatch during matrix multiplication.
#python -m tensorflow.python.debug.examples.debug_errors --error shape_mismatch --debug

# Debugging uninitialized variable.
#python -m tensorflow.python.debug.examples.debug_errors --error uninitialized_variable --debug


# What are the platform-specific system requirements of tfdbg CLI in open-source TensorFlow?
# brew install homebrew/dupes/ncurses

