import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt  
#  k means cluster
import seaborn as sns


cols = ['area', ' perimeter', 'length', 'class','asymmetry']
df = pd.read_csv( ' sample.csv', names = cols, sep = '\s+')

df.head()

for i in range(len(cols) - 1):
    for j in range(i+1, len(cols) - 1):
        x_label = cols[i]
        y_label = cols[j]
        sns.scatterplot(x_label, y_label,data = df, hue = 'class', legend=True)


#  clustering
from sklearn.cluster import KMeans
x = 'perimeter'
y = 'asymmetry '
X = df[x,y].values
kmeans = KMeans(n_clusters=3).fit(X)
clusters = kmeans.labels_
df['class'].values

cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns= [x,y , 'class'])

sns.scatterplot(x, y , hue  = 'class', data = cluster_df)
X = df[cols[:-1]].values    
kmeans = KMeans(n_clusters=3).fit(X)
cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns= df.columns)


#  PCA


from  sklearn.decomposition import PCA

pca = PCA(n_components= 2)
transformed_x = pca.fit_transform(X)
print(transformed_x[:5])
plt.scatter(transformed_x[:, 0 ] ,transformed_x[:, 1 ])
plt.show()

kmeans_pca_df = pd.DataFrame(np.hstack((transformed_x, kmeans.labels_.reshape(-1, 1)    )) , columns= ['pca1' , 'pca2', 'class'])
true_pca_df = pd.DataFrame(np.hstack((transformed_x, df['class'].values.reshape(-1, 1)    )) , columns= ['pca1' , 'pca2', 'class'])
sns.scatterplot(x = 'pca1', y =  'pca2', data = kmeans_pca_df , hue = 'class')
sns.scatterplot(x = x, y = y, data = true_pca_df, hue = 'class')
 