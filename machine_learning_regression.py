import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# for liearn regression, four assumptions: 
# 1. linear        if the graph is linear
# 2. normality          the residual plot, the dots are not clustered
# 3. independence       variables are independent to each other
# 4. homoskedastity     the residual are basically equal, no surge



# evaluate the linear regresion
# 1 mean absolute error (MAE)  mean of absolute residuals
# 2  mean squared error       mean of squarred residuals
# 3 root mean square error      root of mean square error
# 4 coefficients of determination       R2 = 1 -  RSS/TSS   sum of squared residuals  RSS, total sum of  squared mean residuals  TSS

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import seaborn as sbs
import tensorflow as tf  
import copy
from sklearn.linear_model import LinearRegression




df = pd.read_csv(" sample.csv")
dataset_cols = ['bike_count', 'hours', 'temp', 'humidity','wind']
df.drop(['data', 'holidays'], axis = 1)
df.columns = dataset_cols
df['functional'] = (df['functional'] == 'yes').astype(int)
df = df[ df['hours'] == 12]
df.drop(['hours'] , axis = 1)

for i in df.comlumns[1:]:
    plt.scatter(df['labeel'], df['bike_counts'])
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    plt.show()
df.drop(['wind', ' visibility', 'functional'], axis = 1)


# train validation test
train, vali, test = np.split(df.sample(frac = 1), [int(.6* len(df)) , int(.8* len(df))  ])
def get_X_y(df, ylabel, xlabel):
    df1 = df.deepcopy(df)
    if not xlabel :
        x = df[[c for c in df.columns if c != ylabel]].values
    else:
        if len(xlabel) == 1 :
            x = df[xlabel[0]].values.reshape(-1, 1)
        else :
            x = df[xlabel].values


    y = df[ylabels].values.reshape(-1, 1)
    data = np.hstack((x,y))
    return data, x, y
_ , xtrain_, ytrain  = get_X_y(train, 'bike_count',['temp'])
_ , xval, yval  = get_X_y(vali, 'bike_count',['temp'])
_ , xtest, ytest  = get_X_y(test, 'bike_count',['temp'])


regressor = LinearRegression()
regressor.fit(xtrain_, ytrain)
print(regressor.coef_ , regressor.intercept_)
# to see the association
regressor.score(xtest, ytest)   
plt.scatter(xtrain_, ytrain, color = 'blue', label = 'sampel1')
x = tf.linspace(-20, 40, 100)
plt.plot(x, regressor.predicct(x), label  = 'fit', color = 'red', linewidth = 3, label  = 'sample 2 ')
plt.legend()
plt.title('no')
plt.xlabel('')
plt.ylabel('')
plt.show()

#  multiple linear regression
_ , xtrain_all, ytrain_all  = get_X_y(train, 'bike_count',df.columns[1:])
_ , xval_all, yval_all  = get_X_y(vali, 'bike_count',df.columns[1:])
_ , xtest_all, ytest_all  = get_X_y(test, 'bike_count',df.columns[1:])
regressor_all = LinearRegression()
regressor_all.fit(xtrain_all, ytrain_all)
regressor_all.score(xtest_all, ytest_all)

ypredict = regressor_all.predict(xtest_all)






















# regression with neural net
temp_normalizer = tf.keras.layers.Normalization(input_shape = (6, ) , axis = None)
temp_normalizer.adpat(xtrain_.resahpe(-1))
temp_nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(1)

]) 
temp_nn_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=.1,loss = 'mean_squared_error' ))
history = temp_nn_model.fit(
    xtrain_.reshpae(-1), ytrain,
    verbose = 0,
    epochs = 1000, 
    validation_data= (xval, yval )
)
plt.scatter(xtrain_, ytrain, color = 'blue', label = 'sampel1')
x = tf.linspace(-20, 40, 100)
plt.plot(x, temp_nn_model.predicct(x), label  = 'fit', color = 'red', linewidth = 3, label  = 'sample 2 ')
plt.legend()
plt.title('no')
plt.xlabel('')
plt.ylabel('')
plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'], lable = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('binary crossentrophy')
    plt.legend()
    plt.grid(True)
    plt.show()



nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1),

]) 

nn_model.copmile(optimizer = tf.keras.optimizers.Adam(learning_rate=.01) , loss = 'mean_squared_error')

history = nn_model.fit(xtrain_, ytrain  , validation_data = (xval, yval ) , verbose = 0, epochs = 100 , )
plot_loss(history)

plt.scatter(xtrain_, ytrain, color = 'blue', label = 'sampel1')
x = tf.linspace(-20, 40, 100)
plt.plot(x, temp_nn_model.predicct(x), label  = 'fit', color = 'red', linewidth = 3, label  = 'sample 2 ')
plt.legend()
plt.title('no')
plt.xlabel('')
plt.ylabel('')
plt.show()

history = nn_model.fit(xtrain_all, ytrain_all,verbose = 0 ,validation_data = (xval, yval), epochs = 1000 )




#  now copmare the two models , linera model and nureal network
nn_predict = nn_model.predict(xtest_all)

re_predict = regressor.predict(xtest_all)
def MSE(predict, real):

    return (np.square(predict - real)).mean()

print(MSE( nn_predict,ytest_all) )
print(MSE(re_predict , ytest_all ))

plt.axes(aspect =  'equal')
plt.scatter()
plt.xlim= [1,2000  ] 
plt.plot([1,2999 ] , [1,2999 ]  , c = 'red' )
