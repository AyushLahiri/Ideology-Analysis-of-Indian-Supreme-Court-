import pandas as pd 
import numpy as np 
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from transformers import set_seed
import json
import requests 

json_filepath = '.\imp_files\apis\keys.json'

# get api_key
with open(json_filepath, 'r') as file:
    data = json.load(file)
api_key = data['api_keys']['openai']

seed_value = 42
np.random.seed(seed_value)
set_seed(seed_value)

#######################################################################
##################     BERT TOPIC MODELLING       #####################
######################################################## 

df = pd.read_excel(r"D:\ILDC\output_topics.xlsx", usecols= ['name', 'justice_full_names', 'clean_text'])


topic_model = BERTopic(calculate_probabilities=True)

# Fit and transform 
topics, _ = topic_model.fit_transform(df['clean_text'])
topic_info = topic_model.get_topic_info()

# Create a DataFrame mapping documents to topics
document_ids = list(range(len(df['processed_text'])))
doc_topic_mapping = pd.DataFrame({'Document_ID': document_ids, 'Topic': topics})

# Aggregate this mapping with topic information
# This joins the two DataFrames on the topic number
merged_info = doc_topic_mapping.merge(topic_info, left_on='Topic', right_on='Topic')

# Uncomment below to create a new topic info df and the recode it manually. Recommended to load the recoded topic_info_df in next cell
# This is the df that shows us topic number, topic and assigned bag of words representation
topic_info_df = pd.DataFrame(topic_model.get_topic_info())
# We then manually look through this saved df and recode topics and decide which ones to drop
topic_info_df.to_excel(".\Data\topic_info_new_run.xlsx", index=False)

#######################################################################
##################     CHAT GPT TOPIC MODELLING       #################
######################################################## 


# Create a new column with the first 4000 characters of each string in 'text_column'. only the first 40000 characters are fed into gpt
df['first_4000'] = df['clean_text'].str.slice(0, 4000)

def hey_chatGPT(question_text, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-4-turbo",
        "temperature": 0,
        "messages": [{
            "role": "user", 
            "content": question_text
        }]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", 
                             json=data, 
                             headers=headers, timeout=30)
    
    response_json = response.json()
    return response_json['choices'][0]['message']['content'].strip()

# this is our prompt
question_template = """
In this judgement summary text determine if a government agency or the constitution is involved as either plaintiff or defendant. 
If the government or constitution is involved, return the response in the following format: (government role):(specific government agency):(broad issue topic). 
If government is not involved, format your response as: Other: (plaintiff/defendant): (broad issue).

Text: "{}"
"""

#due to cost constraints only keep 2900 rows. sampling 5 years before and after BJP

#extract year from name column 
df['year'] = df['name'].str.slice(0, 4)
sample1 = df[(df['year'] >= 2009) & (df['year'] <= 2013)].sample(n=1400, random_state=42)

# Filter the DataFrame for the second range (2014-2020) and sample 1400 entries
sample2 = df[(df['year'] >= 2014) & (df['year'] <= 2020)].sample(n=1500, random_state=42)

# Concatenate the two samples
final_sample = pd.concat([sample1, sample2], ignore_index=True)



output = []
i = 0  

for text in final_sample.first_4000:
    try:
        formatted_question = question_template.format(text)
        response = hey_chatGPT(formatted_question, key)
        output.append(response)
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        output.append(None)  # Append None or some error indicator
    
    i += 1  # Increment the counter
    if i % 100 == 0:
        print(f"Processed {i} entries")

# Ensure the output length matches the input DataFrame length
print(f"Total processed: {len(output)}, Total rows in df: {len(df)}")

final_sample['topic'] = output

#topic modelled dataset
final_sample.to_excel(r"D:\ILDC\topic_modelled_gpt.xlsx")

