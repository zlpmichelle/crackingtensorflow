
# coding: utf-8

# In[6]:


# predict mediam house values

import tensorflow as tf

feature_column_data = [1,2.4,0,9.9,3,120]
feature_tensor = tf.constant(feature_column_data)

sparse_tensor = tf.SparseTensor(indices=[[0,1], [2,4]],
                              values=[6, 0.5],
                              dense_shape=[3,5])

sess = tf.Session()
sess.run(sparse_tensor)


# In[ ]:


classifier.fit(input_fn=my_input_fn, steps = 2000)

def my_input_function_training_set():
    return my_input_function(training_set)

classifer.fit(input_fn=my_input_fn_training_set, steps=2000)

classifer.fit(input_fn=functools.partial(my_input_function,
                                        data_set=training_set), setps=2000)

classifer.fit(input_fn=lambda: my_input_fn(training_set))

classifer.fit(input_fn=lambda: my_input_fn(test_set), steps=2000)


# In[9]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# In[11]:


COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]

LABEL = "medv"

training_set = pd.read_csv("/Users/lipingzhang/Desktop/program/jd/crackingtensorflow/crackingtensorflow/crackingcode/day5/boston_train.csv", skipinitialspace=True,
                          skiprows=1, names=COLUMNS)

test_set = pd.read_csv("/Users/lipingzhang/Desktop/program/jd/crackingtensorflow/crackingtensorflow/crackingcode/day5/boston_test.csv", skipinitialspace=True,
                      skiprows=1, names=COLUMNS)

prediction_set = pd.read_csv("/Users/lipingzhang/Desktop/program/jd/crackingtensorflow/crackingtensorflow/crackingcode/day5/boston_predict.csv", skipinitialspace=True,
                            skiprows=1, names=COLUMNS)




# In[12]:


features_cols = [tf.contrib.layers.real_valued_column(k)
                for k in FEATURES]


# In[17]:


regressor = tf.contrib.learn.DNNRegressor(feature_columns=features_cols,
                          hidden_units=[10, 10],
                          model_dir="/tmp/boston_model")


# In[18]:


def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values)
                   for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels


# In[20]:


regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)


# In[21]:


ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps = 1)


# In[22]:


loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))


# In[24]:


y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
# .predict() returns an iterator; convert to a list and print predictions
predictions = list(itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))


# In[ ]:




