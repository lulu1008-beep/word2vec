# word2vec
Questions 09/14/21:
All the scripts so far run smoothly, but a number looks suspecious to me. 

The len(model.wv.vocab) shows the size of the new model that was built on the pretrained google news embeddings. 

The Crown corpus (which is modelled after the Brown corpus) is 1 million words, but the len(model.wv.vocab)=3000021. This means it only has 21 new words in the whole Crown corpus. However, I tried to use only one text file to test it, and found the len(model.wv.vocab)=3000013. When comparing the two results, this looks suspecious. 

Could you help me to see if this is normal? And if not, where goes wrong? Thank you! 




Questions 08/25/21:
line 23, in <module>
    model.build_vocab(my_corpus,update=True)

RuntimeError: You cannot do an online vocabulary-update of a model which has no prior vocabulary. First build the vocabulary of your model with a corpus before doing an online update.
[LL] I don't quite understand the error. I checked online, it says set the min_count to 1, rather than the default 5. I actually already did that in line 22. How to solve it? 
  I also tried Pycharm debugging, it showed word2vec(vocab=6) after stepping over line 22. 
 Please heeellllpppp! Many Thanks! 

Questions 08/23/21:
1) errors in the script?
2) looks like I can only get the word vector in my_corpus. If the word is not present in my_corpus, then I can't get its vector. Is this normal? (I think my own model should contain the vectors/vocabulary from the pretrained model and my own model.) 
3) are there only 1323 words in my own model? (Something goes wrong?) 
