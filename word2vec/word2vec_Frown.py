import nltk
import glob
import gensim
import nltk.data
import re
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from nltk import tokenize

nltk.download("punkt")
spl = nltk.data.load('tokenizers/punkt/english.pickle')

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

file_list = []

file_list = glob.glob("./frown.untagged/*.txt")
print(len(file_list))

with open('./Combined_Frown.txt', 'w', encoding='utf8') as outfile:
    for names in file_list:
        # print(names)
        with open(names, 'r', encoding='utf8', errors='ignore') as infile:
            outfile.write(infile.read())
        outfile.write("\n")

with open("./Combined_Frown.txt", 'r', encoding='utf8') as inputfile:
    lines = inputfile.readlines()
b = ''
for line in lines:
    line = line[7:]
    b += line.lstrip()
c = b.replace('\n', ' ')
y1 = ''.join(c)
#print(y1)
y = re.sub(r'\<.*?\>', '', y1)

with open("./Combined_Frown_no_number.txt", 'w', encoding='utf8') as inputfile1:
    inputfile1.write(y)

a = tokenize.sent_tokenize(y)
my_corpus = list(map(lambda x: nltk.word_tokenize(x), a))
# print(my_corpus)

with open("./Combined_Frown_seg.txt", 'w', encoding='utf8') as inputfile2:
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

model.wv.save_word2vec_format("./Frown_vector.bin", binary=True)
print('model is saved.')
model = KeyedVectors.load_word2vec_format("./Frown_vector.bin", encoding='utf8', binary=True)
print('task is done')