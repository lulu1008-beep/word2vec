import nltk
import glob
import gensim
import nltk.data
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from nltk import tokenize
nltk.download("punkt")
spl = nltk.data.load('tokenizers/punkt/english.pickle')

file_list = []

file_list = glob.glob("./Crown_RAW/*.txt")
print(len(file_list))

with open('./Combined_Crown.txt', 'w') as outfile:
    for names in file_list:
        # print(names)
        with open(names, 'r') as infile:
            outfile.write(infile.read())
        outfile.write("\n")

with open("./Combined_Crown.txt", 'r') as text1:
    file1 = text1.read()
    a = tokenize.sent_tokenize(file1)
    my_corpus = list(map(lambda x: nltk.word_tokenize(x), a))
    print('my_corpus is done.')

google_wv = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

model = Word2Vec(size=300, sg=1, min_count=1)
model.build_vocab(my_corpus)
print('initializing is done.')

training_examples_count = model.corpus_count
print('1')
model.build_vocab([list(google_wv.vocab.keys())], update=True)
print('2')
model.intersect_word2vec_format("./GoogleNews-vectors-negative300.bin", binary=True, lockf=1.0)
print('3')
model.train(my_corpus, total_examples=training_examples_count, epochs=model.epochs)
print(len(model.wv.vocab))

model.wv.save_word2vec_format("./Crown_vector.bin", binary=True)
print('model is saved.')
model = KeyedVectors.load_word2vec_format("./Crown_vector.bin", encoding='ISO-8859-1', binary=True)
print('task is done')
