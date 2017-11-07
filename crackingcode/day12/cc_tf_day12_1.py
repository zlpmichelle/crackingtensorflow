
# coding: utf-8

# In[ ]:


# Sequence-to-Sequence Models
#cd models/tutorials/rnn/translate
#python translate.py --data_dir [your_data_directory]

outputs, states = basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)


# In[ ]:


outputs, states = embedding_rnn_seq2seq(
    encoder_inputs, decoder_inputs, cell,
    num_encoder_symbols, num_decoder_symbols,
    embedding_size, output_projection=None,
    feed_previous=False)


# In[ ]:


if num_samples > 0 and num_samples < self.target_vocab_size:
  w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
  w = tf.transpose(w_t)
  b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
  output_projection = (w, b)

  def sampled_loss(labels, inputs):
    labels = tf.reshape(labels, [-1, 1])
    # We need to compute the sampled_softmax_loss using 32bit floats to
    # avoid numerical instabilities.
    local_w_t = tf.cast(w_t, tf.float32)
    local_b = tf.cast(b, tf.float32)
    local_inputs = tf.cast(inputs, tf.float32)
    return tf.cast(
        tf.nn.sampled_softmax_loss(
            weights=local_w_t,
            biases=local_b,
            labels=labels,
            inputs=local_inputs,
            num_sampled=num_samples,
            num_classes=self.target_vocab_size),
        dtype)


# In[ ]:


if output_projection is not None:
  for b in xrange(len(buckets)):
    self.outputs[b] = [tf.matmul(output, output_projection[0]) +
                       output_projection[1] for ...]


# In[ ]:


buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


# In[ ]:


#python translate.py
#  --data_dir [your_data_directory] --train_dir [checkpoints_directory]
#  --en_vocab_size=40000 --fr_vocab_size=40000


# In[ ]:





# In[ ]:


# smaller 

#python translate.py
#  --data_dir [your_data_directory] --train_dir [checkpoints_directory]
#  --size=256 --num_layers=2 --steps_per_checkpoint=50


# In[ ]:


# translate from English to Franch

#python translate.py --decode
#  --data_dir [your_data_directory] --train_dir [checkpoints_directory]

