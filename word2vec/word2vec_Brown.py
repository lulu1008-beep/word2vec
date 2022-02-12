import nltk
import glob
import gensim
import nltk.data
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from nltk import tokenize
nltk.download("punkt")
spl = nltk.data.load('tokenizers/punkt/english.pickle')

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

file_list = []

file_list = glob.glob("./brown.untagged/*.txt")
print(len(file_list))

with open('./Combined_Brown.txt', 'w', encoding='utf8') as outfile:
    for names in file_list:
        # print(names)
        with open(names, 'r') as infile:
            outfile.write(infile.read())
        outfile.write("\n")

with open("./Combined_Brown.txt", 'r', encoding='utf8') as inputfile:
    lines=inputfile.readlines()
b=''
for line in lines:
        line = line[14:]
        b+= line.lstrip()
c=b.replace('\n', ' ')
y=''.join(c)
#print(y)
with open("./Combined_Brown_no_number.txt", 'w', encoding='utf8') as inputfile1:
    inputfile1.write(y)

a = tokenize.sent_tokenize(y)
my_corpus = list(map(lambda x: nltk.word_tokenize(x), a))
#print(my_corpus)
with open("./Combined_Brown_seg.txt", 'w', encoding='utf8') as inputfile2:
    inputfile2.write(str(my_corpus))
print('my_corpus is done.')

google_wv = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

model = Word2Vec(size=300, sg=1, min_count=1)
model.build_vocab(my_corpus)
print('initializing is done.')

training_examples_count = model.corpus_count
model.intersect_word2vec_format("./GoogleNews-vectors-negative300.bin", binary=True, lockf=1.0)
model.train(my_corpus, total_examples=training_examples_count, epochs=model.epochs)
print(len(model.wv.vocab))

model.wv.save_word2vec_format("./Brown_vector.bin", binary=True)
print('model is saved.')
model = KeyedVectors.load_word2vec_format("./Brown_vector.bin", encoding='utf8', binary=True)
print('task is done')