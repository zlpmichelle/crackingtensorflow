
# coding: utf-8

# In[1]:


# Exporting and Importing a MetaGraph

def to_proto(self, export_scope=None):

  """Converts a `Variable` to a `VariableDef` protocol buffer.

  Args:
    export_scope: Optional `string`. Name scope to remove.

  Returns:
    A `VariableDef` protocol buffer, or `None` if the `Variable` is not
    in the specified name scope.
  """
  if (export_scope is None or
      self._variable.name.startswith(export_scope)):
    var_def = variable_pb2.VariableDef()
    var_def.variable_name = ops.strip_name_scope(
        self._variable.name, export_scope)
    var_def.initializer_name = ops.strip_name_scope(
        self.initializer.name, export_scope)
    var_def.snapshot_name = ops.strip_name_scope(
        self._snapshot.name, export_scope)
    if self._save_slice_info:
      var_def.save_slice_info_def.MergeFrom(self._save_slice_info.to_proto(
          export_scope=export_scope))
    return var_def
  else:
    return None

@staticmethod
def from_proto(variable_def, import_scope=None):
  """Returns a `Variable` object created from `variable_def`."""
  return Variable(variable_def=variable_def, import_scope=import_scope)

ops.register_proto_function(ops.GraphKeys.GLOBAL_VARIABLES,
                            proto_type=variable_pb2.VariableDef,
                            to_proto=Variable.to_proto,
                            from_proto=Variable.from_proto)


# In[ ]:


def export_meta_graph(filename=None, collection_list=None, as_text=False):
  """Writes `MetaGraphDef` to save_path/filename.

  Args:
    filename: Optional meta_graph filename including the path.
    collection_list: List of string keys to collect.
    as_text: If `True`, writes the meta_graph as an ASCII proto.

  Returns:
    A `MetaGraphDef` proto.
  """


# In[ ]:


# Build the model

with tf.Session() as sess:
  # Use the model
  ...
# Export the model to /tmp/my-model.meta.
meta_graph_def = tf.train.export_meta_graph(filename='/tmp/my-model.meta')

meta_graph_def = tf.train.export_meta_graph(
    filename='/tmp/my-model.meta',
    collection_list=["input_tensor", "output_tensor"])


# In[ ]:


# export meta graph

...
# Create a saver.
saver = tf.train.Saver(...variables...)
# Remember the training_op we want to run by adding it to a collection.
tf.add_to_collection('train_op', train_op)
sess = tf.Session()
for step in xrange(1000000):
    sess.run(train_op)
    if step % 1000 == 0:
        # Saves checkpoint, which by default also exports a meta_graph
        # named 'my-model-global_step.meta'.
        saver.save(sess, 'my-model', global_step=step)
        
        

# import meta graph

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
  new_saver.restore(sess, 'my-save-dir/my-model-10000')
  # tf.get_collection() returns a list. In this example we only want the
  # first one.
  train_op = tf.get_collection('train_op')[0]
  for step in xrange(1000000):
    sess.run(train_op)


# In[ ]:


# build an inference graph, export it as a meta graph:

# Creates an inference graph.
# Hidden 1
images = tf.constant(1.2, tf.float32, shape=[100, 28])
with tf.name_scope("hidden1"):
  weights = tf.Variable(
      tf.truncated_normal([28, 128],
                          stddev=1.0 / math.sqrt(float(28))),
      name="weights")
  biases = tf.Variable(tf.zeros([128]),
                       name="biases")
  hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
# Hidden 2
with tf.name_scope("hidden2"):
  weights = tf.Variable(
      tf.truncated_normal([128, 32],
                          stddev=1.0 / math.sqrt(float(128))),
      name="weights")
  biases = tf.Variable(tf.zeros([32]),
                       name="biases")
  hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
# Linear
with tf.name_scope("softmax_linear"):
  weights = tf.Variable(
      tf.truncated_normal([32, 10],
                          stddev=1.0 / math.sqrt(float(32))),
      name="weights")
  biases = tf.Variable(tf.zeros([10]),
                       name="biases")
  logits = tf.matmul(hidden2, weights) + biases
  tf.add_to_collection("logits", logits)

init_all_op = tf.global_variables_initializer()

with tf.Session() as sess:
  # Initializes all the variables.
  sess.run(init_all_op)
  # Runs to logit.
  sess.run(logits)
  # Creates a saver.
  saver0 = tf.train.Saver()
  saver0.save(sess, 'my-save-dir/my-model-10000')
  # Generates MetaGraphDef.
  saver0.export_meta_graph('my-save-dir/my-model-10000.meta')


# In[ ]:


# import it and extend it to a training graph.

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
  new_saver.restore(sess, 'my-save-dir/my-model-10000')
  # Addes loss and train.
  labels = tf.constant(0, tf.int32, shape=[100], name="labels")
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat([indices, labels], 1)
  onehot_labels = tf.sparse_to_dense(
      concated, tf.stack([batch_size, 10]), 1.0, 0.0)
  logits = tf.get_collection("logits")[0]
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=onehot_labels, logits=logits, name="xentropy")
  loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")

  tf.summary.scalar('loss', loss)
  # Creates the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(0.01)

  # Runs train_op.
  train_op = optimizer.minimize(loss)
  sess.run(train_op)


# In[ ]:


# Import a graph with preset devices.

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta',
      clear_devices=True)
  new_saver.restore(sess, 'my-save-dir/my-model-10000')
  ...


# In[ ]:


# Import within the default graph.

meta_graph_def = tf.train.export_meta_graph()
...
tf.reset_default_graph()
...
tf.train.import_meta_graph(meta_graph_def)
...


# In[ ]:


# Retrieve Hyper Parameters
filename = ".".join([tf.train.latest_checkpoint(train_dir), "meta"])
tf.train.import_meta_graph(filename)
hparams = tf.get_collection("hparams")

