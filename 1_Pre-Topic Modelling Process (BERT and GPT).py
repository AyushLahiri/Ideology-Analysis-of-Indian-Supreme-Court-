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
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk import pos_tag
import warnings
warnings.filterwarnings('ignore')


### SCRIPT TO CLEAN RAW DATA: EXTRACT JUDGE NAMES, PRE-PROCESS FOR BERT AND GPT TOPIC MODELLING, PRE-PROCESS FOR STM AND ideological scaling. 


df = pd.read_csv(r"D:\ILDC\ILDC_multi\multi.csv")

def extract_justice_names(text):
    # Initialize the stop words set and add court 
    stop_words = ENGLISH_STOP_WORDS.union({'court', 'Court'})

    #names of author justice of the judgement can appea rin many different formats handling that 
    full_name_pattern = r'\b([A-Za-z]+|[A-Za-z]\.?) ([A-Za-z]+),? J\.?\b'
    single_name_pattern = r'\b([A-Za-z]+|[A-Za-z]\.?),? J\.?\b'

    full_name_matches = re.findall(full_name_pattern, text)
    single_name_matches = re.findall(single_name_pattern, text)
    
    filtered_full_names = []
    filtered_single_names = []

    for match in full_name_matches:
        full_name = ' '.join(match).replace(',', '').strip()
        name_parts = full_name.split()
        if not any(part.lower() in stop_words for part in name_parts):
            filtered_full_names.append(full_name)

    for match in single_name_matches:
        single_name = match.replace(',', '').strip()
        if single_name.lower() not in stop_words:
            filtered_single_names.append(single_name)

    
    return filtered_full_names, filtered_single_names

df2 = df.copy()
#extract justice name 
df2[['justice_full_names', 'justice_single_names']] = df2['text'].apply(lambda x: pd.Series(extract_justice_names(x)))
#drop instance if no judgement name available 
df2['justice_full_names'] = df2.apply(lambda row: row['justice_full_names'][0] if len(row['justice_full_names']) > 0 else 'del', axis=1)
df2['justice_single_names'] = df2.apply(lambda row: row['justice_single_names'][0] if len(row['justice_single_names']) > 0 else 'del', axis=1)

#consolidate cases where justice names have only single or double name.  
df2['justice_full_names'] = np.where(df2.justice_full_names == 'del',df2.justice_single_names, df2.justice_full_names)

#drop cases where neither are met 
df2 = df2[df2.justice_full_names != 'del']

#judgement documents generally start with the justice name. In most cases any text appearing before judge name is purely procedural . so we remove that below 

def remove_first_instance(row):
    name_to_remove = row['justice_full_names']
    # Find the first occurrence of the name
    index = row['text'].find(name_to_remove)
    if index != -1:
        if index <= 10:
            # Remove everything up to the end of the first instance of the name
            text = row['text'][index + len(name_to_remove):]
        else:
            # Replace the first occurrence of the name only
            text = row['text'][:index] + row['text'][index + len(name_to_remove):]
    else:
        # If no match is found, use the original text
        text = row['text']
    return text.strip()


df2['new_text'] = df2.apply(remove_first_instance, axis=1)

#next we clean up the text this is done in two rounds 
#here we only remove punctuation, numbers, alphanumeric and words less 2 in length. we essentially keep all gramatically correct instances. 
#the resulting column is then used in BERT topic modelling and chatgpt which require coherent sentences but without noise. 

def clean_text(text):
    import pandas as pd
    import string
    import unicodedata
    from nltk.tokenize import ToktokTokenizer
    from nltk.probability import FreqDist
    import contractions
    import re

    # Expand contractions
    text = contractions.fix(text.lower())
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode("utf-8")
    
    # Replace punctuation with spaces
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    
    # Tokenize text
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    
    # Remove stopwords and filter out short tokens
    tokens = [token for token in tokens if len(token) > 2 and not token.isdigit()]

    tokens = [token for token in tokens if not re.search(r'(?=.*\d)(?=.*[a-zA-Z]).*', token)]
    total_tokens = len(tokens)
    out = ' '.join(tokens)
    return out, total_tokens

df2_2 = df2.copy()
df2_2['clean_text'], df2_2['token_count'] = zip(*df2_2['new_text'].apply(clean_text))

token_counts = df2_2['token_count']
summary_descriptives = {
    'mean': token_counts.mean(),
    'min':token_counts.min(),
    'max':token_counts.max(),
    'median': token_counts.median(),
    'std_dev': token_counts.std(),
    'total': token_counts.sum()}

#this is going to be used as input for BERT and chat gpt topic modelling 
df2_2.to_excel(r".\output_for_topics.xlsx")


## in this round, we take all the afforementioned steps and create a new column but this time remove stopwords

def clean_text_aggresive(text):
    import pandas as pd
    import string
    import unicodedata
    from nltk.tokenize import ToktokTokenizer
    from nltk.probability import FreqDist
    import contractions
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    import re

    # Expand contractions
    text = contractions.fix(text.lower())
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode("utf-8")
    
    # Replace punctuation with spaces
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    
    # Tokenize text
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    
    # Remove stopwords and filter out short tokens
    tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS and len(token) > 2 and not token.isdigit()]

    tokens = [token for token in tokens if not re.search(r'(?=.*\d)(?=.*[a-zA-Z]).*', token)]
    
    return ' '.join(tokens)

df3 = df2.copy()
df3['clean_text_stopword_removed'] = df3['new_text'].parallel_apply(clean_text)

#this is going to be used in R to find a list of most frequent and rarest words. and in turn will be used in STM. 
#in the next pre-processing script we also remove the frequent and rare words and then use this column for ideological scaling. 
# thus the first round of cleaning is solely for BERT and gpt topic modelling and no other purposes

df4 = df3[['name', 'justice_full_names', 'clean_text_stopword_removed']]
df4.to_excel(r"D:\ILDC\send_inR_find_freq_rare_words.xlsx")
