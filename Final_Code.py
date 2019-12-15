#!/usr/bin/env python
# coding: utf-8

#BASED ON THE EXPERIMENTS IN THE JUPYTER NOTEBOOK, GOING AHEAD WITH LIGHT GBM REGRESSION MODEL


# # Rating Suggestor
# ## The aim is to build a model which predicts the possible rating when a user enters text into the review space 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

#Reading data
Data=pd.read_csv('Data.csv')

# ## Since Natural Language Processing stage took a lot of time to process all the records, taking only a subset of records to build the model, acknowledging that accuracy will be lesser

#Review ID and Date seem like unnecessary variables, dropping review_id and date
Data=Data.drop(['review_id','date'],axis=1)


# ## Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder

# Encoding user_id
le2=LabelEncoder()
Data['user_id']=le2.fit_transform(Data['user_id'])
Data['user_id'].unique()


# Encoding business_id
le3=LabelEncoder()
Data['business_id']=le3.fit_transform(Data['business_id'])
Data['business_id'].unique()

# ## Natural Language Processing on the Review Text
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Text Cleaning: turning to lower case, removing non-alphanumeric characters and 
corpus=[]
for i in range(len(Data['text'])):
    review=re.sub('[^a-zA-Z]',' ',Data['text'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english','french'))]
    review=' '.join(review)
    print(review)
    corpus.append(review)

	
#Creating a bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()

#Creating a count vector for the top 10000 most frequent words in the corpus
A=cv.fit_transform(corpus).toarray()
A=pd.DataFrame(A)

#Adding the count vector to the input matrix
Data=pd.concat([Data,A], axis=1)

#Removing raw text column
Data=Data.drop('text',axis=1)


# ## Separating input and output features
y=Data['stars']
# ### Scaling the numerical data (funny, cool and useful), to tackle outliers 
from scipy.stats import zscore
Data.iloc[:, 3:6]=Data.iloc[:, 3:6].apply(zscore)
x=Data.drop('stars',axis=1)

# ## Splitting Training and Test datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=7)

#Importing Metrics
from sklearn.metrics import mean_squared_error

# ## Training a LightGBM Model (Chosen because it is a fast, tree based ensemble method)
import lightgbm as lgb
# ## Using LightGBM Regression
lgb_train = lgb.Dataset(xtrain, ytrain)
lgb_eval = lgb.Dataset(xtest, ytest, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'root_mean_squared_error',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                early_stopping_rounds=750)

predictedlgbm2 = gbm.predict(xtest, num_iteration=gbm.best_iteration)
predictedlgbm2



# # Limitations and Additional Tasks
# ### 1. Memory failure due to very large dataset. Given sufficient infrastructure a more detailed and accurate model can be created
# ### 2. LightGBM has been used, but other model alternatives were Naive Bayes and Random Forest (comprison of the three models can be done. Even neural networks can be used if sufficient memory available
# ### 3. For Data Preprocessing, instead of the count vector, even Term Frequency - Inverse Document Frequency (TF-IDF), or Cosine similarity could be explored (Cosine Similarity could be a better choice)
