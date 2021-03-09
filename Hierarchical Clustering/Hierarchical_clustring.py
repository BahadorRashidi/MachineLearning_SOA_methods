#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
# %matplotlib inline

data = pd.read_csv('Wholesale customers data.csv')
print(data.head())

data_scaled = pd.DataFrame(normalize(data), columns=data.columns)

plt.plot(data_scaled)
plt.show()

#%% Here we apply the hierarchical clustring and find the linkage Z and plot th dendograme
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
Z = shc.linkage(data_scaled, method='ward')
dend = shc.dendrogram(Z)

'''
The result of this snipet is that we conclude two major clusters for the given dataset.
Then we would like to use these two clusters to see visually how much of milk vs grocery
belong to each cluster. With this aim we use the folloiwng code snipet 
'''
#%%
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data_scaled)


plt.figure(figsize=(10, 7))  
plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_) 

