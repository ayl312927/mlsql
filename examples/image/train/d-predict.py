from __future__ import absolute_import
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

model = SVC()
model2 = MultinomialNB()
print(hasattr(model,"partial_fit"))
print(hasattr(model2,"partial_fit"))
