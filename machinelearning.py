import numpy as np
import pandas as pd



# k nearest method
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(x_train, y_train)
y_prediction = knn_model.predict(x_test)
y_prediction

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model  = nb_model.fit(x,y)

print(classification_report(y_train, y_predict))