import gensim
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
from collections import namedtuple
import logging
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from nltk.tokenize import word_tokenize
from pandarallel import pandarallel
import tqdm
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
pandarallel.initialize(progress_bar=True,nb_workers=8)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
assert gensim.models.doc2vec.FAST_VERSION > -1


##### THIS SCRIPT CONTAINS ALL THE SCALING METHODS AND OUTPUTS. TAKES AS INPUT THE OUTPUT FROM THE SECOND PROCESSING SCRIPT #######

df  = pd.read_excel(r"D:\ILDC\final_df.xlsx") 

#################################################################################################################

################### DOC2VEC with PCA (adapted from Rheault and Cochrane(2019)) ###################################

##################################################################################################################

#corpus iterator adapted from Rheault and Cochrane (2019), to append document tags. each document tag is unique judge name 
class corpusIterator:
    def __init__(self, df, bigram=None, trigram=None):
        self.df = df
        if bigram:
            self.bigram = bigram
        else:
            self.bigram = None
        if trigram:
            self.trigram = trigram
        else:
            self.trigram = None
    def __iter__(self):
        self.speeches = namedtuple('speeches', 'words tags')
        for row in self.df.iterrows():
            text = row[1]['filtered_text'].replace('\n','')
            name = str(row[1]['justice_full_names'])
            year = row[1]['year']
            justicetag = name
            tokens = text.split()
            if self.bigram and self.trigram:
                self.words = self.trigram[self.bigram[tokens]]
            elif self.bigram and not self.trigram:
                self.words = self.bigram[tokens]
            else:
                self.words = tokens
            self.tags = [justicetag]
            yield self.speeches(self.words, self.tags)
class phraseIterator():
    def __init__(self, df):
        self.df = df
    def __iter__(self):
        for row in self.df.iterrows():
            text = row[1]['filtered_text'].replace('\n','')
            yield text.split()

#train doc2vec model and save 
model0 = Doc2Vec(vector_size=200, window=10, min_count=50, workers=8, epochs=5)
model0.build_vocab(corpusIterator(df))
model0.train(corpusIterator(df), total_examples=model0.corpus_count, epochs=model0.epochs)
model0.save(r'D:\ILDC\model')

model = Doc2Vec.load(r"D:\ILDC\model")


#Interpretor class adapted from Rheault and Cochrane(2019) to find word embeddings most similar to first two principal components of document embeddings 
class Interpret(object):
    def __init__(self, model, parties, dr, Z, rev1=False, rev2=False, min_count=100, max_count = 1000000, max_features=10000):
            self.model = model
            self.parties = parties
            self.P = len(self.parties)
            self.M = self.model.vector_size
            self.voc = self.sorted_vocab(min_count, max_count, max_features)
            self.V = len(self.voc)
            self.pca = dr
            self.max = Z.max(axis=0)
            self.min = Z.min(axis=0)
            self.sims = self.compute_sims()
            self.dim1 = rev1
            self.dim2 = rev2

    def sorted_vocab(self, min_count=100, max_count=1000000, max_features=10000):
        wordlist = []
        for word in self.model.wv.index_to_key:  # Use index_to_key to iterate over words
            count = self.model.wv.get_vecattr(word, "count")  # Use get_vecattr to get the word frequency
            wordlist.append((word, count))
        wordlist = sorted(wordlist, key=lambda tup: tup[1], reverse=True)
        return [w for w, c in wordlist if c > min_count and c < max_count][0:max_features]

    def compute_sims(self):

        S = np.zeros((self.V, 2))
        for idx, w in enumerate(self.voc):
            S[idx, :] = self.pca.transform(self.model.wv[w].reshape(1,-1))
        sims_right = euclidean_distances(S, np.array([self.max[0],0]).reshape(1, -1))
        sims_left = euclidean_distances(S, np.array([self.min[0],0]).reshape(1, -1))
        sims_up = euclidean_distances(S, np.array([0,self.max[1]]).reshape(1, -1))
        sims_down = euclidean_distances(S, np.array([0,self.min[1]]).reshape(1, -1))
        temp = pd.DataFrame({'word': self.voc, 'right': sims_right[:,0], 'left': sims_left[:,0], 'up': sims_up[:,0], 'down': sims_down[:,0]})
        return temp

    def top_words_list(self, topn=20, savepath='D:/ILDC/table1.txt'):

        with open(savepath, 'w') as f:
            print("Table 1: Interpreting PCA Axes", file=f)
            if self.dim1:
                ordering = ['left','right']
            else:
                ordering = ['right', 'left']
            temp = self.sims.sort_values(by=ordering[0])
            print(80*"-", file=f)
            print("Words Associated with Positive Values (Right) on First Component:", file=f)
            print(80*"-", file=f)
            self.top_positive_dim1 = temp.word.tolist()[0:topn]
            self.top_positive_dim1 = ', '.join([w.replace('_',' ') for w in self.top_positive_dim1])
            print(self.top_positive_dim1, file=f)
            temp = self.sims.sort_values(by=ordering[1])
            print(80*"-", file=f)
            print("Words Associated with Negative Values (Left) on First Component:", file=f)
            print(80*"-", file=f)
            self.top_negative_dim1 = temp.word.tolist()[0:topn]
            self.top_negative_dim1 = ', '.join([w.replace('_',' ') for w in self.top_negative_dim1])
            print(self.top_negative_dim1, file=f)

            if self.dim2:
                ordering = ['down','up']
            else:
                ordering = ['up', 'down']
            temp = self.sims.sort_values(by=ordering[0])
            print(80*"-", file=f)
            print("Words Associated with Positive Values (North) on Second Component:", file=f)
            print(80*"-", file=f)
            self.top_positive_dim2 = temp.word.tolist()[0:topn]
            self.top_positive_dim2 = ', '.join([w.replace('_',' ') for w in self.top_positive_dim2])
            print(self.top_positive_dim2, file=f)
            temp = self.sims.sort_values(by=ordering[1])
            print(80*"-", file=f)
            print("Words Associated with Negative Values (South) on Second Component:", file=f)
            print(80*"-", file=f)
            self.top_negative_dim2 = temp.word.tolist()[0:topn]
            self.top_negative_dim2 = ', '.join([w.replace('_',' ') for w in self.top_negative_dim2])
            print(self.top_negative_dim2, file=f)
            print(80*"-", file=f)

#conduct PCA
document_ids = range(len(model.dv)) 
document_vectors = np.array([model.dv[i] for i in document_ids])

pca = PCA(n_components=2)
principal_components = pca.fit_transform(document_vectors)

#bring in format required by class Interpretor 
parties = [d for d in model.dv.key_to_index ]
M = model.vector_size; P = len(parties)
z = np.zeros((P,M))
for i in range(P):
    z[i,:] = model.dv[parties[i]]
pca = PCA(n_components = 2)
Z = pd.DataFrame(pca.fit_transform(z), columns = ['dim1', 'dim2'])

#outputs table of axes extremes and most associated words by euclidean distance
Interpret(model, parties, pca, Z, rev1=False, rev2=False, min_count=100, max_count = 1000000, max_features = 50000).top_words_list(20)

#creating PCA dataframe for plotting 
plot2 = Z.copy()
plot2['docid'] = parties
grouped = df.groupby('justice_full_names').agg({
    'year':'first'})
grouped.reset_index(inplace = True)
plot2 = pd.merge(plot2,grouped, how = 'left', left_on = 'docid', right_on = 'justice_full_names')
plot2 = plot2 [['dim1', 'dim2', 'docid', 'year']]

labels_to_highlight = ['y chandrachud', 'bobde', 'kishan kaul', 'markandey katju', 'kurian joseph']
colors = ['blue' if year < 2014 else 'orange' for year in plot2['year']]
plt.figure(figsize=(8, 6))
plt.scatter(plot2['dim1'], plot2['dim2'], alpha=0.5, c=colors)
plt.savefig('./figure_1.png')

import random
for idx, row in plot2.iterrows():
    if row['docid'] in labels_to_highlight:
        if row['docid'] == 'bobde':
            x_jitter = random.uniform(-1, 1)  # Adjust the jitter range as needed
            y_jitter = random.uniform(-1, 1)
        else:
            x_jitter = random.uniform(0.5, 0.5)  # Adjust the jitter range as needed
            y_jitter = random.uniform(0.5, 0.5)  # Adjust the jitter range as needed
        plt.text(row['dim1'] + x_jitter, row['dim2'] + y_jitter, row['docid'], fontsize=9, fontweight='bold')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Document Vectors')
plt.grid(True)
plt.savefig('./figure_2.png')

#time series plot of each principal component 

plt.legend(handles=[blue_patch, orange_patch])

plot2 = plot2.sort_values(by='year')

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)


axes[0].plot(plot2['year'], plot2['dim1'], marker='o', linestyle='-', color='blue')
axes[0].set_title('Principal Component 1 Over Time')
axes[0].set_ylabel('Principal Component 1')


axes[1].plot(plot2['year'], plot2['dim2'], marker='o', linestyle='-', color='green')
axes[1].set_title('Principal Component 2 Over Time')
axes[1].set_ylabel('Principal Component 2')
axes[1].set_xlabel('Year')

for ax in axes:
    ax.grid(True)

plt.tight_layout()
plt.savefig('./figure_3.png')

#########################################################################################################

######################### PoliticalBiasBert available at HugginFace ######################################

##########################################################################################################

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
def find_ideology(text):
    text_entry = text
    tokens = word_tokenize(text_entry)
    #tokens limited to first 512 tokens. begin input only from 500th token to ensure we are getting meat of the document with respect to argumentation
    last_512_tokens = tokens[500:]
    final_text = ' '.join(last_512_tokens)
    text = final_text

    inputs = tokenizer(text, return_tensors="pt", truncation = True)
    
    labels = torch.tensor([0])
    outputs = model(**inputs, labels=labels)
    loss, logits = outputs[:2]
    
    probabilities = logits.softmax(dim=-1)[0].tolist() 
    sorted_indices = np.argsort(probabilities)

    first_highest_idx = sorted_indices[-1]
    second_highest_value = probabilities[sorted_indices[-2]]
    first_highest_value = probabilities[first_highest_idx]

    #documents most likely to be labelled centrist due to document. If second highest probabiltiy is 0.15 or closer to first highest and first highest is centrist 
    ##then use second highest as the idelogy (indicating center-left, center-right)

    if first_highest_value - second_highest_value <= 0.15:
        second_highest_idx = sorted_indices[-2]
    else:
        second_highest_idx = first_highest_idx
    return first_highest_idx, second_highest_idx

from tqdm.auto import tqdm
tqdm.pandas(desc="progress!")
df['ideology'] = df['filtered_text'].progress_apply(find_ideology)

df[['preds_first', 'preds_second']] = pd.DataFrame(df['ideology'].tolist(), index=df.index)
def served_under_category(years):
    pre_bjp_years = [y for y in years if y < 2014]
    return 'pre_bjp' if len(pre_bjp_years) > len(years) / 2 else 'post_bjp'

# Group by 'justice_name' and apply the 'served_under_category' function
df['served_under'] = df.groupby('justice_full_names')['year'].transform(served_under_category)

df2 = df.copy()

#group judges to find average ideology 
group_judges = df2.groupby('justice_full_names').agg({
    'year':'first',
    'served_under': 'first',  # or 'last' if you want to retain the last entry
    'preds_first': 'mean',
    'preds_second': 'mean'
})
group_judges.reset_index(inplace = True)

group_judges['preds_first'] = group_judges.preds_first.round(3)
group_judges['preds_second'] = group_judges.preds_second.round(3)

group_judges.sort_values(by = 'preds_first', inplace = True)

#merge with grouped to get year 
group_judges = pd.merge(group_judges,grouped, how = 'left', on = 'justice_full_names')
group_judges['served_under'] = np.where(group_judges.year < 2014, 'pre-bjp', 'post-bjp')
group_judges.to_csv('D:/ILDC/ideologies.csv')

#plot results 
blue_patch = mpatches.Patch(color='blue', label='Pre-BJP Nomination')
orange_patch = mpatches.Patch(color='orange', label='Post-BJP Nomination')
labels_to_highlight = ['y chandrachud', 'bobde', 'kishan kaul',
       'markandey katju', 'kurian joseph']
colors = ['blue' if s == 'pre-bjp' else 'orange' for s in group_judges['served_under']]

plt.figure(figsize=(10, 8))
plt.scatter(group_judges['year'].values,group_judges['preds_first'].values, alpha=0.5, c=colors)

# Loop through the DataFrame rows
for idx, row in group_judges.iterrows():
    if row['justice_full_names'] in labels_to_highlight:
        plt.text(row['year'], row['preds_first'], row['justice_full_names'], fontsize=9, fontweight='bold' )

plt.xlabel('Year')
plt.ylabel('Ideological Score')
plt.title('Ideology of Justices (2009-2020)')


plt.grid(True)

plt.savefig('./figure_4.png')