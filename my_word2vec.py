import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
google_wv = gensim.models.KeyedVectors.load_word2vec_format('/Users/lulubanana/Desktop/GoogleNews-vectors-negative300.bin', binary=True)

# Init blank english spacy nlp object
import spacy
nlp = spacy.blank('en')

keys = []
for idx in range(3000000):
    keys.append(google_wv.index_to_key[idx])

# Set the vectors for our nlp object to the google news vectors
nlp.vocab.vectors = spacy.vocab.Vectors(data=google_wv.vectors, keys=keys)

print(nlp.vocab.vectors.shape)
#(3000000, 300)

my_corpus=[["first", "sentence"],["second", "sentence"],["third", "sentence"], ["fourth", "sentence"], ["chink", "sentence"]]

model = Word2Vec(vector_size=300, min_count=1, epochs=10)
model.build_vocab(my_corpus,update=True)
training_examples_count = model.corpus_count
model.build_vocab(list(google_wv.index_to_key), update=True)
model.intersect_word2vec_format("/Users/lulubanana/Desktop/GoogleNews-vectors-negative300.bin",binary=True, lockf=1.0)
model.train(my_corpus,total_examples=training_examples_count, epochs=model.epochs)
print(len(model.wv.key_to_index))