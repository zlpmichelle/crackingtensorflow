
# coding: utf-8

# In[ ]:


# Recurrent Neural Networks

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
state = tf.zeros([batch_size, lstm.state_size])
probabilities = []
loss = 0.0
for current_batch_of_words in words_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(current_batch_of_words, state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)


# In[ ]:


# Placeholder for the inputs in a given iteration.
words = tf.placeholder(tf.int32, [batch_size, num_steps])

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
initial_state = state = tf.zeros([batch_size, lstm.state_size])

for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = lstm(words[:, i], state)

    # The rest of the code.
    # ...

final_state = state


# In[ ]:


# A numpy array holding the state of LSTM after each batch of words.
numpy_state = initial_state.eval()
total_loss = 0.0
for current_batch_of_words in words_in_dataset:
    numpy_state, current_loss = session.run([final_state, loss],
        # Initialize the LSTM state from the previous iteration.
        feed_dict={initial_state: numpy_state, words: current_batch_of_words})
    total_loss += current_loss


# In[ ]:


# embedding_matrix is a tensor of shape [vocabulary_size, embedding size]
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)


# In[ ]:


lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=False)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * number_of_layers,
    state_is_tuple=False)

initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = stacked_lstm(words[:, i], state)

    # The rest of the code.
    # ...

final_state = state


# In[ ]:


# download http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
# to /tmp/simple-examples/data/
#cd /tmp
#wget -r -np -nd http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
#tar zxf simple-examples.tgz

#cd ~/Desktop/program/jd/tensorflow_models/models/tutorials/rnn/ptb/cd ~/Desktop/program/jd/tensorflow_models/models/tutorials/rnn/ptb/
#nohup python ptb_word_lm.py --data_path=/tmp/simple-examples/data/ --model=small > 1102_rnn.log 2>&1 & 

