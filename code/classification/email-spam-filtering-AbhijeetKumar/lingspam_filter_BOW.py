
# coding: utf-8

"""
@author: M.Amintoosi
"""

import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer


def read_corpus(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    corpus = []
    for file_path in files:
        with open(file_path) as f_input:
            lines = f_input.readlines()
            # دو خط اول و دوم را نمی‌خواهیم
            corpus.append(lines[2].rstrip('\n'))
    return corpus


train_dir = 'lingspam_public/train-mails'
train_docs = read_corpus(train_dir)

# Prepare feature vectors per training mail and its labels
vect = CountVectorizer()
vect.fit(train_docs)

train_X = vect.transform(train_docs)
train_labels = [fi.startswith('spm') for fi in os.listdir(train_dir)]


# Training SVM and Naive bayes classifier and its variants

model1 = LinearSVC()
model2 = MultinomialNB()

model1.fit(train_X,train_labels)
model2.fit(train_X,train_labels)


# Test the unseen mails for Spam

test_dir = 'lingspam_public/test-mails'
test_docs = read_corpus(test_dir)
test_X = vect.transform(test_docs)
test_labels = [fi.startswith('spm') for fi in os.listdir(test_dir)]

result1 = model1.predict(test_X)
result2 = model2.predict(test_X)

print(confusion_matrix(test_labels,result1))
print(confusion_matrix(test_labels,result2))

