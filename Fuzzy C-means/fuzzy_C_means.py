#%%
import pandas as pd
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal


#%%

df = pd.read_csv('Iris.csv')
df.drop(['Id'], axis=1, inplace=True)

columns = df.columns.to_list()
labels = df['Species']
df = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]



#%%
# Number of Clusters
k = 3
# Maximum number of iterations
MAX_ITER = 100
# Number of data points
n = len(df)
# Fuzzy parameter
m = 1.7 #Select a value greater than 1 else it will be k-means





#%%






#%%




#%%








