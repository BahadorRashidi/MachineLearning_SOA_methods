
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

#%% Here we apply clustring on the residual index that has been created
from scipy.cluster.hierarchy import dendrogram, linkage
X = np.zeros(data_scaled.shape) 
X = data_scaled.to_numpy()
#%%
# generate the linkage matrix
Z = linkage(X, 'ward')

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        # plt.figure(figsize=(30,30))
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

# set cut-off to 50
max_d = 2.5  # max_d as in max_distance
fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()
#%%
from scipy.cluster.hierarchy import fcluster
max_d = 2.5
clusters = fcluster(Z, max_d, criterion='distance')

## OR if we know how many clusters we have
# k=2

# fcluster(Z, k, criterion='maxclust')

plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
plt.show()


