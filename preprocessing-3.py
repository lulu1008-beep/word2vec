import nltk
nltk.download("punkt")
import nltk.data
spl=nltk.data.load('tokenizers/punkt/english.pickle')

import glob
file_list = []

file_list = glob.glob("./CLOB_RAW/*.txt")
print(len(file_list))

with open('./Combined_CLOB_RAW.txt', 'w') as outfile:
    for names in file_list:
        #print(names)
        with open(names,'r') as infile:
            outfile.write(infile.read())
        outfile.write("\n")

def process_text(file):
    text=[]
    x=spl.tokenize(file)
    for sent in x:
        tokens=nltk.word_tokenize(sent)
        text.append(tokens)
    return text

with open ("./Combined_CLOB_RAW.txt", 'r') as text1:
    file1=text1.read()
with open("./segmented_combined_CLOB_RAW.txt", 'a+') as g:
        #for value in process_text_file (file):
        s=process_text(file1)
        g.write(str(s))

import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
google_wv = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
with open("./segmented_combined_CLOB_RAW.txt", 'r') as gg:
    my_corpus=gg.read()
    #print(my_corpus)
model = Word2Vec(size=300, sg=1,min_count=1)
model.build_vocab(my_corpus) #,update=True
training_examples_count = model.corpus_count
#model.build_vocab(list(google_wv.index_to_key), update=True)
model.build_vocab([list(google_wv.vocab.keys())], update=True)
model.intersect_word2vec_format("./GoogleNews-vectors-negative300.bin",binary=True, lockf=1.0)
model.train(my_corpus,total_examples=training_examples_count, epochs=model.epochs)
print(len(model.wv.vocab))

model.wv.save_word2vec_format("./CLOB_Raw_vector.bin", binary=True)
print('model is saved.')
model=KeyedVectors.load_word2vec_format("./CLOB_Raw_vector.bin", encoding='ISO-8859-1', binary=True)
print('task is done')


