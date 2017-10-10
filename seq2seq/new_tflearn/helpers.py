import numpy as np

def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """
    
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    
    return inputs_time_major, sequence_lengths

def loadDataFile(feature=None):
    file = open(feature, "r")
    lines = []
    with file as myFile:
        for line in file:
            feat = []
            line = line.split(' ')
            for i in range(0, len(line)):
                feat.append(int(line[i]))
            lines.append(feat)

    while True:
        yield lines

import random
def random_sampler(filename, k):
    samples = []
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()

        random_set = sorted(random.sample(range(filesize), k))

        for i in range(k):
            f.seek(random_set[i])
            # Skip current line (because we might be in the middle of a line)
            f.readline()
            # Append the next line to the sample set
            li = f.readline().rstrip()
            if li == '':
                break

            feat = []
            line = li.split(' ')
            for i in range(0, len(line)):
                feat.append(int(line[i]))
            samples.append(feat)

    '''
    print(len(samples))
    for i in samples:
        print(i)
    '''
    return samples



def random_next_batch_k(features_filename, labels_filename, k, filesize):
    random_set = sorted(random.sample(range(filesize), k))
    batches_features = []
    with open(features_filename, 'r') as ff:
        ff.seek(0)
        #print("random:" + str(random_set))
        for n, line in enumerate(ff):
            #print("**" + str(n))
            if int(n) in random_set:
                li = line.rstrip()
                if li == '':
                    break
                feat = []
                line = str(li).split(' ')
                for i in range(0, len(line)):
                    feat.append(int(line[i]))
                batches_features.append(feat)
            else:
                continue

    #print("***" + str(batches_features))

    batches_labels = []
    with open(labels_filename, 'r') as fl:
        fl.seek(0)
        for n, line in enumerate(fl):
            #print("**" + str(n))
            if int(n) in random_set:
                li = line.rstrip()
                if li == '':
                    break
                feat = []
                line = str(li).split(' ')
                for i in range(0, len(line)):
                    feat.append(int(line[i]))
                batches_labels.append(feat)
            else:
                continue
    #print("***" + str(batches_labels))
    return batches_features, batches_labels



def next_batch_k(filename, k, lastoffset):
    batches = []
    with open(filename, 'r') as f:
        f.seek(0)
        start = lastoffset
        end = lastoffset + k - 1
        for n, line in enumerate(f):
            if n < start:
                continue
            elif n >= start and n <= end:
                li = line.rstrip()
                if li == '':
                    break
                feat = []
                line = str(li).split(' ')
                for i in range(0, len(line)):
                    feat.append(int(line[i]))
                batches.append(feat)
            else:
                break
    return batches



def loadData(feature=None, length=100):
    file = open(feature, "r")
    lines = []
    with file as myFile:
        for line in file:
            feat = []
            line = line.split(',')
            for i in range(length - 1):
                feat.append(float(line[i]))
            feat.append(float(line[length-1][:-1]))
            lines.append(feat)
        
    while True:
        yield lines

def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]
