import pandas as pd
from gensim.models import KeyedVectors


model = KeyedVectors.load_word2vec_format("./Brown_vector.bin", encoding='utf8', binary=True)

whites_new=list(set([''])) # provide a list of White keywords
blacks = list(set([''])) # provide a list of Black keywords
hispanics = list(set([''])) # provide a list of Hispanic keywords
asians = list(set(['']))# provide a list of Asian keywords


read_file = pd.read_csv('./top 100/Brown_LC_new.txt')
read_file.to_csv('./top 100/Brown_LC_1.csv', index=None)
cols = []
with open('./top 100/Brown_LC_1.csv', 'r') as csvfile:
    df = pd.read_csv(csvfile, header=None)
    headerList = ['att', 'sim_score']
    df.to_csv("./top 100/Brown_LC_1.csv", header=headerList, index=False)
    csvfile2 = pd.read_csv("./top 100/Brown_LC_1.csv")
    # print(csvfile2)
    saved_col = csvfile2['att']
    for row in saved_col:
        cols.append(row[2:-1])
    print(cols)
cols_new = list(set(cols))

x = [] # sim
y = [] # ethnic group
z = [] # warmth/competence attributes

for att in cols_new:
# todo "asians'
    for country in whites_new:
        try:
            currentSim = model.wv.similarity(att, country)
            x.append(currentSim)
            y.append(country)
            z.append(att)
        except KeyError:
            results = 0
df = pd.DataFrame({'quality': z, 'people': y, 'sim': x})

print(df)
# todo
df.to_csv('./top 100/Brown_LC_NewW_sim_modified.csv')
df2 = df.groupby('quality').mean()
print(df2)
# todo
df2.to_csv('./top 100/Brown_LC_NewW_sim_groupby_modified.csv')

