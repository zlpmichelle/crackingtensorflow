
# coding: utf-8

# In[26]:


#load.py
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import gensim
import os
import re
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument

def get_doc_list(folder_name):
    doc_list = []
    file_list = [folder_name + "/" + name for name in os.listdir(folder_name) if name.endswith('txt')]
    for file in file_list:
        st = open(file, 'r').read()
        doc_list.append(st)
    print('Found %s documents under the dir %s ..... ' % (len(file_list), folder_name))
    return doc_list

def get_doc(folder_name):
    doc_list = get_doc_list(folder_name)
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    
    taggeddoc = []
    
    texts = []
    for index, i in enumerate(doc_list):
        # for tagged doc
        wordslist = []
        tagslist = []
        
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        
        # remove stop words from tokens
        stopped_tokens = [i  for i in tokens if not i in en_stop]
        
        # remove numbers
        number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
        number_tokens = ' '.join(number_tokens).split()
        
        #stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
        #remove empty
        length_tokens = [i for i in stemmed_tokens if len(i) > 1]
        #add tokens to list
        texts.append(length_tokens)
        
        td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(), str(index))
        taggeddoc.append(td)
        
    return taggeddoc
    
    


# In[27]:


sentence = TaggedDocument(words = [u'some', u'words', u'here'], tags = [u'SEN_1'])

print sentence


# In[28]:


# doc2vectest.py
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import codecs
import gensim
#import load
#documents = load.get_doc('docs')
documents = get_doc('/Users/lipingzhang/Desktop/program/doc2vec/word_vectors_game_of_thrones-LIVE/data')
print('Data Loading finished')
print(len(documents), type(documents))

# build the model
model = gensim.models.Doc2Vec(documents, dm = 0, alpha = 0.025, size = 20, min_alpha = 0.025, min_count = 0)

# start training
for epoch in range(200):
    if epoch % 20 == 0:
        print('Now training epoch %s' & epoch)
    model.train(documents)
    # decrease the learning rate
    model.alpha -= 0.002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

# shows the similar words
print(model.most_similar('suppli'))

# shows the learnt embeeding
print(model['suppli'])

# shows the similar docs with id = 2
print(model.doc2vecs.most_similar(str(2)))


# In[29]:


model.most_similary('suppli')


# In[30]:


model['suppli']


# In[31]:


model.doc2vecs.most_similar(str(2))
model.save('save/trained.model')
model.save_word2vec_format('save/trained.word2vec')


# In[32]:


# load the word2vec
word2vec = gensim.models.Doc2Vec.load_word2vec_format('save/trained.word2vec')
print(word2vec['good'])

# load the doc2vec
model = gensim.models.Doc2Vec.load('save/trained.model')
docvecs = model.docvecs
#print(docvecs[str(3)])


# In[33]:


def plotWords():
    # get model, we can use w2v only
    w2v, d2v = useModel()

    words_np = []
    # a list of labels (words)
    words_label = []
    for word in w2v.vocab.keys():
        words_np.append(w2v[word])
        words_label.append(word)

    print('Added %s words. Shape %s' % (len(words_np), np.shape(words_np)))


    pca = decomposition.PCA(n_components = 2)
    pca.fit(words_np)
    reduced = pca.transform(words_np)

    # plt.plot(pca.explained_variance_ratio)
    for index,vec in enumerate(reduced):
        # print('%s %s' % (words_label[index], vec))
        if index <100:
            x,y = vec[0], vec[1]
            plt.scatter(x, y)
            plt.annotate(words_label[index], xy=(x,y))
    plt.show()


# In[ ]:




