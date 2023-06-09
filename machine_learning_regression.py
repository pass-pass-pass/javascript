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