
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicity
print(node1, node2)


# In[3]:


sess = tf.Session()
print(sess.run([node1, node2]))


# In[5]:


node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3):", sess.run(node3))


# In[6]:


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)


# In[7]:


print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))


# In[9]:


add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))


# In[10]:


W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b


# In[12]:


init = tf.global_variables_initializer()
sess.run(init)


# In[13]:


print(sess.run(linear_model, {x: [1,2,3,4]}))


# In[15]:


y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


# In[16]:


fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


# In[17]:


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


# In[18]:


sess.run(init) # reset values to incorrect defaults
for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
    
print(sess.run([W, b]))


# In[21]:


# complete progame
import numpy as np
import tensorflow as tf

# model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of squares 

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})
    
# evaluation training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s" %(curr_W, curr_b, curr_loss))


# In[27]:


# high lever API tf.contrib.learn

import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension = 1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns= features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1, -2, -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
print(estimator.evaluate(input_fn=input_fn))



# In[30]:


# a custom model
import numpy as np
import tensorflow as tf
# Declare list of features, we only have one real-valued feature
def model(features, lables, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", [1], dtype = tf.float64)
    b = tf.get_variable("b", [1], dtype = tf.float64)
    y = W * features['x'] + b
    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - lables))
    # traing sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
   
    # ModelFnOps connects subgraphs we built to the
    # appropriate functionality.
    return tf.contrib.learn.ModelFnOps(mode = mode, predictions = y, loss = loss, train_op = train)

estimator = tf.contrib.learn.Estimator(model_fn = model)
# define our data set
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({'x': x}, y, 4, num_epochs = 1000)

# train
estimator.fit(input_fn = input_fn, steps = 1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps = 10))


# In[ ]:




