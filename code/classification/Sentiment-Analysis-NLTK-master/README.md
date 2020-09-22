# Sentiment-Analysis-NLTK
Sentiment Analysis / Opinion Mining for provided data in NLTK corpus using NaiveBayesClassifier Algorithm

## https://github.com/sachin-bisht/Sentiment-Analysis-NLTK

## Accuracy
Without using bigrams : 72.8

With using bigrams : 85.0
Accuracy improved using bigram.

### M.Amintoosi
https://www.pythonforengineers.com/build-a-sentiment-analysis-app-with-movie-reviews/
...
For each word, we create a dictionary with all the words and True. Why a dictionary? So that words are not repeated. If a word already exists, it won’t be added to the dictionary.

Let’s see how this works:

create_word_features(["the", "quick", "brown", "quick", "a", "fox"])
 
{'brown': True, 'fox': True, 'quick': True}

We call our function with the string “the quick brown quick a fox”.

You can see that a) The stop words are removed  b) Repeat words are removed  c) There is a True with each word.

Again, this is just the format the Naive Bayes classifier in nltk expects.