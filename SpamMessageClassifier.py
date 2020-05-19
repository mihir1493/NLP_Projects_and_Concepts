# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:59:25 2020

@author: LENOVO
"""

import pandas as pd
import sklearn
messages = pd.read_csv('SpamClassifierDataset/SMSSpamCollection', sep = '\t',
                       names= ["Label", "message"])

import re
import nltk
#nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()
corpus = []


for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages["message"][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()     
    
y=  pd.get_dummies(messages["Label"])
y = y.iloc[:,1].values    
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
