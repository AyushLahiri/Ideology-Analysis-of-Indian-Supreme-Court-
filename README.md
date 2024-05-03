# Analyzing Ideology and Emotion: A Textual Examination of Indian Supreme Court Justices

## Project Overview
This project aims to investigate the ideological diversity among Supreme Court judges in India, particularly in cases involving governmental participation. By analyzing oral judgments given by judges across all cases they presided over within a year, we seek to uncover latent attributes indicative of judicial ideology. This process involves using zero-shot topic modeling with ChatGPT, followed by experimentation with different scaling methodologies, including Doc2Vec and pre-trained transformers.

### Required Input Dataset
The dataset required for this project is available upon request at the following GitHub repository: [Exploration-Lab/CJPE](https://github.com/Exploration-Lab/CJPE). 

## Tutorial Workflow

### 1. Pre-Topic Modeling Process (BERT and GPT)
- **Objective**: Process raw ILDC data.
- **Steps**:
  - Extract names of presiding judges.
  - Generate two sets of outputs based on different text preprocessing approaches:
    - **Output 1**: Serves as input for "3_Topic Modeling".
    - **Output 2**: Feeds into "2_remove_freq_words_STM".

### 2. Remove Frequent Words (STM)
- **Functions**:
  - Generate a list of the 90% most common and most rare words as a DataFrame.
  - Conduct Structural Topic Modeling (STM).

### 3. Topic Modeling (BERT and GPT)
- **Description**: Implement BERT-based topic modeling and ChatGPT-driven topic identification.
- **Requirements**: OpenAI API key.
- **Outputs**: 
  - 2,935 randomly sampled documents, including columns for ChatGPT-based topics, as a CSV file.
  - BERT topic outputs and bag of words representations for each topic.

### 4. Pre-Process Final DataFrame
- **Process**:
  - Remove the most common and most rare words.
  - Standardize names of judges to ensure uniqueness.
  - Exclude cases with unclear or unavailable judge names.
- **Output**: Cleaned data as a CSV file for further analysis.

### 5. Scaling Ideological Bias
- **Tools**: Doc2Vec and PoliticalBiasBERT.
  - **Doc2Vec**: Employing PCA using the methodology outlined by Rheault and Cochrane (2019).
  - **PoliticalBiasBERT**: Explore ideological scaling using the model available at [Hugging Face's PoliticalBiasBERT](https://huggingface.co/bucketresearch/politicalBiasBERT).
- **Results**: Output the results of both scaling methods and provide interpretations of the PCA axis for Doc2Vec.

## Output Interpretation
The results will be visualized and interpreted to understand the potential biases and ideological leanings of the judges based on the textual analysis of their judgments.