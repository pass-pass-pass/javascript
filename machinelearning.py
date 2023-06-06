import numpy as np
import pandas as pd




# k nearest method
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(x_train, y_train)
y_prediction = knn_model.predict(x_test)
y_prediction

from sklearn.naive_bayes import GaussianNB


nb_model = GaussianNB()
nb_model  = nb_model.fit(x,y)

print(classification_report(y_train, y_predict))


#logistic regression

from sklearn.linear_model import LogisticRegression


lg_regression_model = LogisticRegression()
lg_model  = lg_regression_model.fit(x,y) 
y_prediction = lg_model.predict(x_test)
print(classification_report(y_prediction, y_true))



#support vector machine


from sklearn.svm import SVC
svm_model = SVC()
svm_model = svm_model.fit(x_train,y_train)
y_predict = svm_model.predict(x_test)
print(classification_report(y_true, y_predict))



train, valid, test = np.split(df.sample(frac = 1), [int(.6* len(df), int(.8 * len(df)))])
def plot_loss(history):
    plt.plot(history.history['loss'], lable = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('binary crossentrophy')
    plt.legend()
    plt.grid(True)
    plt.show()

import tensorflow as tf

def plot_history(history):
    fig, (ax1,ax2)= plt.subplot(1,2)
    ax1.plot(history.history['loss'], label = 'loss')
    ax1.plot(history.history['val_loss'], label = 'val_loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('binary crossentropy')
    ax1.grid(True)
    ax2.plot(history.history['accuracy'], label = 'accuracy')
    ax2.plot(history.history['val_accuracy'], label= 'val_accuracy')
    ax2.set_xlabel('epoach')
    ax2.set_ylabel('accuracy')
    ax2.grid(True)
    plt.show()


def train_model(x_train,y_train, num_nodes,dropout_prob,learning_rate,batch_size, epochs ):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape = (10,) ,),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob ),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    nn_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate), loss = 'binary_crossentrophy', metrics = ['accuracy'])
    history = nn_model.fit(x_train, y_train, batch_size, epochs, validation_split= 0.2)
    plot_loss(history)
    return nn_model, history
epoch = 100
least_val_loss = float('inf')
least_model = None

for node in [16,32, 64]:
    for dropout in [0,0.02]:
        for learning_rate in [.005, .001, .01, .1]:
            for batch_size in [32,64,128]:
                print(f"{node} nodes, {dropout} prob , {learning_rate} learning_rate, {batch_size} batch_size")
                model, history = train_model(x_train,y_train, node,dropout,learning_rate,batch_size, epoch)
                val_loss = model.evaluate(x_valid, y_valid)
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_model = model
                plot_loss(history)
                plot_history(history)

