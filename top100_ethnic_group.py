from gensim.models import KeyedVectors

#todo
model = KeyedVectors.load_word2vec_format("./Crown_vector.bin", encoding='utf8', binary=True)

whites_words = []
blacks_words = []
hispanics_words = []
asians_words = []

wordlist = ['Whites', 'Blacks', 'Hispanics', 'Asians']

if 'whites' in model.wv.vocab:
    whites_words.append(model.wv.most_similar(positive='whites', topn=100))
    with open("./Crown_whites_Group.txt", 'w') as text:
        for item in whites_words:
            for w_word in item:
                text.write(str(w_word) + '\n')
    print(whites_words)
else:
    print("this word is not found in the trained model.")

print()

if 'Blacks' in model.wv.vocab:
    blacks_words.append(model.wv.most_similar(positive='Blacks', topn=100))
    with open("./Crown_Blacks_Group,.txt", 'w') as text:
        for b_item in blacks_words:
            for b_word in b_item:
                text.write(str(b_word) + '\n')
    print(blacks_words)
else:
    print("this word is not found in the trained model.")

print()

if 'Hispanics' in model.wv.vocab:
    hispanics_words.append(model.wv.most_similar(positive='Hispanics', topn=100))
    with open("./Crown_Hispanics_Group.txt", 'w') as text:
        for h_item in hispanics_words:
            for h_word in h_item:
                text.write(str(h_word) + '\n')
    print(hispanics_words)
else:
    print("this word is not found in the trained model.")

print()

if 'Asians' in model.wv.vocab:
    asians_words.append(model.wv.most_similar(positive='Asians', topn=100))
    with open("./Crown_Asians_Group.txt", 'w') as text:
        for a_item in asians_words:
            for a_word in a_item:
                text.write(str(a_word) + '\n')
    print(asians_words)
else:
    print("this word is not found in the trained model.")