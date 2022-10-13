# imdb pt-br sentiment analysis: Project overview

**Tags: NLP, logistic regression, tf-idf, AUC, ROC, bag of words, word cloud, stop words, stemming, confusion matrix, sentiment analysis**  

This notebook is based on Alura's course [Linguagem Natural parte 1: NLP com análise de sentimento](https://cursos.alura.com.br/course/nlp-com-analise-de-sentimento) (Natural Language Processing Part 1: Sentiment analysis). Here you will find the bag of words and TF-IDF to count terms in the corpus, word clouds to visualize key terms, and logistic regression to predict negative and positive reviews. The main steps were:

- EDA to understand the data.
- Label encoded the target variable from object to boolean. 
- Created a bag of words to count words.
- Removed punctuations, stop words, and diacritical signs, and stemmed the terms.
- Build two logistic regression (LR) models with the non-cleaned data as the baseline and with cleaned ones.
- Used TF-IDF to measure important terms in the corpus.
- Used AUC metric to evaluate the models. 
- Plotted word clouds for positive and negative reviews to see the most used words for both classes.

## Code and resources

Platform: Jupyter Notebook

Python version: 3.7.6

Packages: NLTK, matplotlib, pandas, numpy, seaborn and sklearn

## Data set

The data set is based on Internet Movie Data Base (IMDB) reviews in English translated into Portuguese. It contains around 100k positive and negative reviews from IMDB, all stored in 4 columns.  

**Data set URL: https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr**

## Model building

- Built two logistic regression models with non-cleaned data as the baseline and with cleaned ones, using the bag of words to count the terms.
- Lastly, created another LR model using the TF-IDF technique.

## Model performance

- The baseline model had an AUC of 0.715 for non-cleaned data and 0.767 for cleaned texts which was a huge improvement.
- Using the TF-IDF, the model has an astonishing AUC score of 0.955, which could be due to overfitting.
