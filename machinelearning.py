import numpy as np
import pandas as pd




# k nearest method
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# knn_model = KNeighborsClassifier(n_neighbors=1)
# knn_model.fit(x_train, y_train)
# y_prediction = knn_model.predict(x_test)
# y_prediction

# from sklearn.naive_bayes import GaussianNB


# nb_model = GaussianNB()
# nb_model  = nb_model.fit(x,y)

# print(classification_report(y_train, y_predict))


# #logistic regression

# from sklearn.linear_model import LogisticRegression


# lg_regression_model = LogisticRegression()
# lg_model  = lg_regression_model.fit(x,y) 
# y_prediction = lg_model.predict(x_test)
# print(classification_report(y_prediction, y_true))



# #support vector machine


# from sklearn.svm import SVC
# svm_model = SVC()
# svm_model = svm_model.fit(x_train,y_train)
# y_predict = svm_model.predict(x_test)
# print(classification_report(y_true, y_predict))

import tensorflow as tf
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape = (10,) ,),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])


nn_model.compile(optimizer= tf.keras.optimizers.Adam)