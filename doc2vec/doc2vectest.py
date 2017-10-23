# doc2vectest.py
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

#import codecs
import gensim
import load

documents = load.get_doc('/Users/lipingzhang/Desktop/program/doc2vec/word_vectors_game_of_thrones-LIVE/data')
#documents = get_doc('/Users/lipingzhang/Desktop/program/doc2vec/word_vectors_game_of_thrones-LIVE/data')
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


model.most_similary('suppli')
model['suppli']

model.doc2vecs.most_similar(str(2))

model.save('save/trained.model')
model.save_word2vec_format('save/trained.word2vec')


# load the word2vec
word2vec = gensim.models.Doc2Vec.load_word2vec_format('save/trained.word2vec')
print(word2vec['good'])

# load the doc2vec
model = gensim.models.Doc2Vec.load('save/trained.model')
docvecs = model.docvecs
#print(docvecs[str(3)])


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


#https://ireneli.eu/2016/07/27/nlp-05-from-word2vec-to-doc2vec-a-simple-example-with-gensim/