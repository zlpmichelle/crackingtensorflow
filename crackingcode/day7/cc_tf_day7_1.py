
# coding: utf-8

# In[2]:


#Variables: Creation, Initialization, Saving, and Loading
import tensorflow as tf
weight = tf.Variable(tf.random_normal([784,200], stdev=0.35), name="weights")
sess = tf.Variable(tf.zeros([200]), name="biases")


# In[3]:


# Pin a variable to CPU.
with tf.device("/cpu:0"):
  v = tf.Variable(...)

# Pin a variable to GPU.
with tf.device("/gpu:0"):
  v = tf.Variable(...)

# Pin a variable to a particular parameter server task.
with tf.device("/job:ps/task:7"):
  v = tf.Variable(...)


# In[ ]:


# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
...
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Later, when launching the model
with tf.Session() as sess:
  # Run the init operation.
  sess.run(init_op)
  ...
  # Use the model
  ...


# In[5]:


Variables: Creation, Initialization, Saving, and Loading

Contents
Creation
Device placement
Initialization
Initialization from another Variable
Custom Initialization
Saving and Restoring
Checkpoint Files
Saving Variables
Restoring Variables
Choosing which Variables to Save and Restore

When you train a model, you use variables to hold and update parameters. Variables are in-memory buffers containing tensors. They must be explicitly initialized and can be saved to disk during and after training. You can later restore saved values to exercise or analyze the model.

This document references the following TensorFlow classes. Follow the links to their reference manual for a complete description of their API:

The tf.Variable class.
The tf.train.Saver class.
Creation

When you create a Variable you pass a Tensor as its initial value to the Variable() constructor. TensorFlow provides a collection of ops that produce tensors often used for initialization from constants or random values.

Note that all these ops require you to specify the shape of the tensors. That shape automatically becomes the shape of the variable. Variables generally have a fixed shape, but TensorFlow provides advanced mechanisms to reshape variables.

# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
Calling tf.Variable() adds several ops to the graph:

A variable op that holds the variable value.
An initializer op that sets the variable to its initial value. This is actually a tf.assign op.
The ops for the initial value, such as the zeros op for the biases variable in the example are also added to the graph.
The value returned by tf.Variable() value is an instance of the Python class tf.Variable.

Device placement

A variable can be pinned to a particular device when it is created, using a with tf.device(...): block:

# Pin a variable to CPU.
with tf.device("/cpu:0"):
  v = tf.Variable(...)

# Pin a variable to GPU.
with tf.device("/gpu:0"):
  v = tf.Variable(...)

# Pin a variable to a particular parameter server task.
with tf.device("/job:ps/task:7"):
  v = tf.Variable(...)
N.B. Operations that mutate a variable, such as tf.Variable.assign and the parameter update operations in a tf.train.Optimizer must run on the same device as the variable. Incompatible device placement directives will be ignored when creating these operations.

Device placement is particularly important when running in a replicated setting. See tf.train.replica_device_setter for details of a device function that can simplify the configuration for devices for a replicated model.

Initialization

Variable initializers must be run explicitly before other ops in your model can be run. The easiest way to do that is to add an op that runs all the variable initializers, and run that op before using the model.

You can alternatively restore variable values from a checkpoint file, see below.

Use tf.global_variables_initializer() to add an op to run variable initializers. Only run that op after you have fully constructed your model and launched it in a session.

# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
...
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Later, when launching the model
with tf.Session() as sess:
  # Run the init operation.
  sess.run(init_op)
  ...
  # Use the model
  ...


# In[6]:


# Create a variable with a random value.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
# Create another variable with the same value as 'weights'.
w2 = tf.Variable(weights.initialized_value(), name="w2")
# Create another variable with twice the value of 'weights'
w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")


# In[ ]:


# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  ..
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in file: %s" % save_path)


# In[ ]:


# Note that when you restore variables from a file you do not have to initialize them beforehand.

# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...


# In[ ]:


# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore only 'v2' using the name "my_v2"
saver = tf.train.Saver({"my_v2": v2})
# Use the saver object normally after that.
...


# In[ ]:




