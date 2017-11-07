
# coding: utf-8

# In[ ]:


# Large-scale Linear Models with TensorFlow

eye_color = tf.contrib.layers.sparse_column_with_keys(
  column_name="eye_color", keys=["blue", "brown", "green"])

education = tf.contrib.layers.sparse_column_with_hash_bucket(    "education", hash_bucket_size=1000)


# In[ ]:


sport = tf.contrib.layers.sparse_column_with_hash_bucket(    "sport", hash_bucket_size=1000)
city = tf.contrib.layers.sparse_column_with_hash_bucket(    "city", hash_bucket_size=1000)
sport_x_city = tf.contrib.layers.crossed_column(
    [sport, city], hash_bucket_size=int(1e4))


# In[ ]:


age = tf.contrib.layers.real_valued_column("age")


# In[ ]:


age_buckets = tf.contrib.layers.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])


# In[ ]:


e = tf.contrib.learn.LinearClassifier(feature_columns=[
  native_country, education, occupation, workclass, marital_status,
  race, age_buckets, education_x_occupation, age_buckets_x_race_x_occupation],
  model_dir=YOUR_MODEL_DIRECTORY)
e.fit(input_fn=input_fn_train, steps=200)
# Evaluate for one step (one pass through the test data).
results = e.evaluate(input_fn=input_fn_test, steps=1)

# Print the stats for the evaluation.
for key in sorted(results):
    print("%s: %s" % (key, results[key]))


# In[ ]:


e = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=YOUR_MODEL_DIR,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])

