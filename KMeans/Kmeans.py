#Importing the libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Height_Weight.csv")
data.head()
x=data.values
sc = StandardScaler()
x=sc.fit_transform(x)
kmeans=KMeans(n_clusters=4)
cluster=kmeans.fit_predict(x)
print(cluster)
#Within clusters sum of squares wcss
wcss=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) # The value of within sum of squares
plt.plot(range(1,10),wcss)
plt.title("Elbow Method")
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()



