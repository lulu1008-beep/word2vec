from gensim.models import KeyedVectors

#todo
model = KeyedVectors.load_word2vec_format("./Crown_vector.bin", encoding='utf8', binary=True)
def top100(wordlist):
    similar_words=[]
    for word in wordlist:
        if word in model.wv.vocab:
            similar_words.append(model.wv.most_similar(positive=word, topn=100))
    final_dict = {}

   # de-duplicate words
    for i in range(len(similar_words)):
        for word_list in similar_words[i]:
            if word_list[0] not in final_dict.keys():
                final_dict[word_list[0]] = word_list[1]
            else:
                if word_list[1] > final_dict[word_list[0]]:
                    final_dict[word_list[0]] = word_list[1]

    deduplicated_list = list(final_dict.items())
    return deduplicated_list

words_w = ['unkind','unfriendly']
words_c = ['incompetence','incompetent']
top100(words_w)
print(top100(words_w))
# print(len(top100(words_w))


#todo: 2 parts
with open("./top100_Crown_negW.txt", 'w', encoding='utf8') as text1:
    for item in top100(words_w):
        for similar in item:
            text1.write(str(similar) +'\n')
    # print(len(str(similar)))

   
