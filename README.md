# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Subreddit Text Classification

## Project 3 - Subreddit Text Classification Model

### Details
Name: Damar Shipp Jr

Class: DSB-10155



### This project aims to classify Reddit posts into two subreddit: 'dogs', and 'personalfinance'; which have been divided into two main parts: 
1. **Data Collection and Cleaning:** Extracts data from Reddit and clean it up: Identifying null values, dropping or replacing null values, combining datasets, etc.. from Reddit.
1. **Model Building:** Preprocess data, train and evaluate machine learning models to classify post.


---


# Table of Contents

1. [Exective Summary](#exective-summary)
2. [Purpose](#purpose)
3. [Problem Statement](#problem_statement)
4. [Results](#results)
5. [Data Overview](#data_overview)
6. [Model Explanation](#model_explanation)
7. [Next Steps](#next_steps)
8. [Requirements](#requirements)

---
## Exective Summary

- Classification of text data is a critical in natural language processing (NLP), especially for large platforms like Reddit. Reddit is where posts must be correctly categorized for effective community engagement and moderation. This project will focus on developing a classsification model to classify posts between two subreddits: dogs_subreddit and personalfinance_subreddit. Using logistic regression and the random forest model, the project will demonstrate how supervised learning can be utilized to address the challenge of categorization within a large platform. The model's performance was evaluated by accuracy, precision, and recall whic make up the classification report. Along with confusion matrices, providing insight into both interpretability and the power of predictiveness.

---
## Purpose

- The primary goal of this project is to build a text classification model that can accurately predict which of two subreddits a given post belongs to. This capability is to help researchers by automating the category of content, improving effiency.

---

## Problem Statement

- Reddit is comprised of numerous communities known as subreddits, where each subreddit is tailored to a specific interest, and misclassified posts can disrupt community discussions and reduce user engagement. To address this; I have taliored this project to develop and evaluate two supervised learning models that will determine which subreddit the post belongs to.

| Model Type             | Model Description|  
|------------------------|-------------------------------------------------------|  
| Logistic Regression    | A baseline model offering simplicity and interpretability. |  
| Random Forest          | A model capturing non-linear relationships for better accuracy. |


- The success of the model is measured by evaluating accuracy, precision, recall, and F1-score on the test dataset. By streamlining subreddit classification, this project directly benefits moderators and researchers.

---
## Results

- After setting up the model; I ran a test on the data to see how well the models were performing and I was excited to see how the models performmed. It highlighted being 1 step closer to satisfying the problem statement: "...misclassified posts can disrupt community discussions and reduce user engagement I have taliored this project to develop and evaluate two supervised learning models." Below are the reults from the models...

| Metric                  | Logistic Regression | Random Forest  |  
|-------------------------|---------------------|----------------|  
| Test Set Accuracy       | 99.83%                | 98.01%            |  
| Precision (dogs_subreddit) | 1.00               | 1.00           |  
| Recall (dogs_subreddit)    | 1.00               | 0.96           |  
| F1-Score (dogs_subreddit)  | 1.00               | 0.98           |  
| Precision (personalfinance_subreddit) | 1.00                | 0.96           |  
| Recall (personalfinance_subreddit)    | 1.00               | 1.00           |  
| F1-Score (personalfinance_subreddit)  | 1.00               | 0.98           |

- Model on New data outside of the normal dataset.


| Metric                          | Logistic Regression       | Random Forest            |  
|-------------------------------|---------------------------|--------------------------|  
| Test Accuracy       | 99.48%            | 96.99%

---

## Data Overview 
#### Provided Data

#### Source: Posts were collected using the Reddit API from two subreddits: dogs_subreddit and personalfinance_subreddit.
- Features: text (preprocessed with lemmatization and stopword removal).
- Target labels: [1] (dogs_subreddit), [0] (personalfinance_subreddit).
- Training Data Size: 1402 subreddit posts.
- Test Data Size: 600 subreddit posts.
##### New data collection:
- New Dataset: A new fetch of unseen data for validation purposes.
- Test Data size 2000 subreddit post
---
## Model Explanation
#### Methods

##### Data cleaning:
1. Identify null values
1. Drop any missing values(Due to high data availability; we dropped the 9 rows of missing values)
##### Preprocessing:
- Text was preprocessed with:
1. Lowercasing.
1. Removing non-alphabetic characters.
1. [Lemmatization](Lemmatization) and [stopword removal](stopword_removal).
1. Vectorized using TF-IDF to create numerical features.
##### Models:
1. Logistic Regression: Chosen for its simplicity and interpretability. It assigns probabilities to the labels and is ideal for identifying significant features.
1. Random Forest: An ensemble method that handles non-linear relationships and improves robustness by aggregating predictions from multiple decision trees.
Evaluation Metrics:
1. Confusion matrix, accuracy, precision, recall, and F1-score were used to measure model performance.
---
## Next Steps
- Multi-class Classification: Extend the model to classify posts across more subreddits.
- Hyperparameter Tuning: Optimize model parameters for possibly better performance.
- Real-Time Classification: Deploy the model in a web application for real-time post categorization.
---
## Requirements
To retrace my steps, ensure you have the following...

- Python
  
- Pandas

- Praw

- NLTK (nltk word_tokenize, stop_words, wordNetLemmatizer)

- Counter
  
- Numpy
  
- Matplotlib
  
- Seaborn

- Scikit Learn(Linear Regression, TTS, Dummy Regressor, CountVectorizer, TfidfVectorizer, RandomForestClasiifier, accuracy_score, classification_report, confusion_matrix)

- **API Access:** Reddit API credentials for data collection. (Get API Acess but viewing Praw setup folder)

#### Contact
- [Damar Shipp Jr LinkedIn](www.linkedin.com/in/damar-shipp-jr-614b71186)
