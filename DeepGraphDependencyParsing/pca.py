from sklearn.decomposition import PCA
import numpy as np
import json
import matplotlib.pyplot as plt
import csv

#Reading the json file
infile = open('output/model_pos_embeddings.json')
model_embeddings = infile.read()
infile.close()

json_data = json.loads(model_embeddings)
word_json = json_data['word']
pos_json = json_data['pos']

words = []
words_values = []
pos = []
pos_values = []
present = {}

#Reading words, pos and their arrays
for word in word_json:
    words.append(word)
    words_values.append(word_json[word])
for p in pos_json:
    pos.append(p)
    pos_values.append(pos_json[p])

#Apply principal component analysis
word_np = np.array(words_values)
pos_np = np.array(pos_values)
pca = PCA(n_components = 2)
pca.fit(word_np.T)
word_pca = pca.components_.T
pca = PCA(n_components = 2)
pca.fit(pos_np.T)
pos_pca = pca.components_.T

with open('data/english/train.conll', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) > 0:
            if row[3] == "VERB":
                present[row[1]] = row[3]

plot_words = []
plot_words_values = []
for i in range(len(words)):
    #if words[i] not in present:
        #pass
    if words[i] in present:
        plot_words.append(words[i])
        plot_words_values.append(word_pca[i,:].tolist())

fig = plt.figure()
figure, axes = plt.subplots()
axes.scatter(np.array(plot_words_values)[:,0], np.array(plot_words_values)[:,1])
for i in range(len(plot_words)):
    axes.annotate(plot_words[i], (np.array(plot_words_values)[i, 0],
    np.array(plot_words_values)[i, 1]))
plt.show()
figure.savefig('word_visualisation.png')

fig = plt.figure()
figure, axes = plt.subplots()
axes.scatter(pos_pca[:,0], pos_pca[:,1])
for i in range(len(pos)):
    axes.annotate(pos[i], (pos_pca[i,0], pos_pca[i, 1]))
plt.show()
figure.savefig('pos_visualization.png')
