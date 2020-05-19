# Kaggle Challenge - Fake News
## Build a system to identify unreliable news articles
*	Develop a machine learning program to identify when an article might be fake news
**Dataset URL:** https://www.kaggle.com/c/fake-news/

## Steps:
*	Importing Data 
*	Stemming the words using Porter Stemmer
*	Using Count Vectorizer to convert the filtered data into an array
*	Splitting the data
*	Modelling using Passive Aggressive, MultinomialNB Algorithms on the dataset 

### MultinomialNB Classifier Algo
![alt text](https://github.com/mihir1493/NLP_Projects_and_Concepts/blob/master/FakeNewsClassifier/Capture_m.JPG "MultinomialNB Classifier Algo")
Results:
accuracy:   0.900
Confusion matrix, without normalization

### Passive Agressive Classifier Algo
![alt text](https://github.com/mihir1493/NLP_Projects_and_Concepts/blob/master/FakeNewsClassifier/Capture_p.JPG "MultinomialNB Classifier Algo")
Results:
accuracy:   0.921
Confusion matrix, without normalization
 
### Multinomial With hyperparameter tuning
Performing hyperparameter tuning on the data, we obtain the best result using the following parameter:
Results:
Alpha: 0.1, Score: 0.9009113504556753

### Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, nltk, itertools
