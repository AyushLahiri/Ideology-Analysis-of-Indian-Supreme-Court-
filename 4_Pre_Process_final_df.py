import pandas as pd 
import numpy as np 
from nltk.tokenize import ToktokTokenizer
import string
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from functools import reduce
import pandas as pd
import unicodedata
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True,nb_workers=8)
import sys
import re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')


#From R, we find the most frequent and rarest words and import as a list 
removed_words_df = pd.read_csv(r'D:\ILDC\removed_words.csv')

#This contains the topic modelled set 
df_topic = pd.read_excel(r"D:\ILDC\topic_modelled_gpt.xlsx", usecols=['name', 'text', 'year' ,'label'])

#the rawest dataset 
df_original = pd.read_csv(r"D:\ILDC\ILDC_multi\multi.csv")

removed_words_list = removed_words_df['x'].tolist() 

#we remove the most frequent and rare words 
def filter_text(text):
    import pandas as pd
    removed_words_df = pd.read_csv('D:/ILDC/removed_words.csv')
    removed_words_list = removed_words_df['x'].tolist()
    removed_words_list+= ['appeal', 'appeals']
    tokens = text.split()
    tokens = [token for token in tokens if token not in removed_words_list]
    return ' '.join(tokens)

#this was the df with the filtered text column with the more aggressive cleaning. from here we remove the frequency words 
df_processed = pd.read_excel(r"D:\ILDC\send_inR_find_freq_rare_words.xlsx")
df_processed['filtered_text'] = df_processed['clean_text'].parallel_apply(filter_text)

# we merge with rows in topic model to have the fully processed text and tpic of only those documents which have modelled 
df_comb = pd.merge(df_topic,df_processed, how = 'left', on = 'name')

#we append in the original dataframe, to have the original text in hand useful for later interpretation and investigation 
df_comb = pd.merge(df_comb,df_original, how = 'left', on = 'name')
df_comb = df_comb[['name', 'justice_full_names','year','text_y','filtered_text', 'label_x']]
df_comb.rename(columns = {'label_x':'label'}, inplace = True)

#from the topics we then extract the rows which have judgements related to the government, by subsetting out non governmental cases
df_comb[['role', 'role2', 'issue']] = df_comb['label'].str.split(':', n=2, expand=True)

df_comb = df_comb[~df_comb.role.str.contains('Other')]
df_comb = df_comb[~df_comb.role2.str.contains('Other')]

#filtered_text is fully processed text column to be used in scaling 
df_comb = df_comb[['name', 'justice_full_names','year','text_y','filtered_text', 'label']]

#next we eject names that are unclear 
drop = ['order sheet', 'castigating dutta', 'function near','brother sathasivam', 'jeffrey','sri','v','k','individual cases','marshall','ist reason',
'patricia', 'ansari respondent', 'dave','powell','new delhi','m s', 'darling','india versus','shri']

#and manually comb through the remaining names to normalize all judge names 
rename = {'rastogi':'ajay rastogi', 
'agrawal':'k agrawal',
'ist kurian':'kurian joseph',
'kurian':'kurian joseph',
's radhakrishnan': 'brother radhakrishnan',
'judgment banumathi':'banumathi',
'a bobde':'bobde',
'sinha':'b sinha',
'prasad': 'k prasad',
'kumar ganguly': 'Ashok Ganguly',
'ganguly': 'Ashok Ganguly',
'sirpurkar': 's sirpurkar',
'harold':'dipak misra'}

df_comb['justice_full_names'] = df_comb['justice_full_names'].str.lower().str.strip()
df_comb['justice_full_names'] = df_comb['justice_full_names'].str.replace(r'[^\w\s]', '', regex=True)
df_comb['justice_full_names'] = df_comb['justice_full_names'].replace(rename)
df_comb = df_comb[~df_comb['justice_full_names'].isin(drop)]

#standardizing the names through a dictionary, reverse dictionary method 
contains_dict = {}
unique_names = df_comb['justice_full_names'].unique()
for name in unique_names:
    if " " not in name: 
        contains_dict[name] = [full_name for full_name in unique_names if name in full_name]

reversed_dict = {}
for key, values in contains_dict.items():
    for value in values:
        if value in reversed_dict:
            reversed_dict[value].append(key)
        else:
            reversed_dict[value] = [key]

filtered_dict = {k: v[0] for k, v in reversed_dict.items() if k != v[0]}
final_dict = {}
for key,values in filtered_dict.items():
    final_dict[values] = key

#judge names are now normalized 
df_comb['justice_full_names'] = df_comb['justice_full_names'].replace(final_dict)

#final dataset to be used in scaling 
df_comb.to_excel(r"D:\ILDC\final_df.xlsx")