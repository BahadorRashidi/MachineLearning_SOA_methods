#%%
import skfuzzy as fuzz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Some styling
sns.set_style("white")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.figsize'] = (8,7)




#%% Generate some data
# Define three cluster centers
centers = [[1, 3],
           [2, 2],
           [3, 8]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.1, 0.5],
          [0.5, 0.3],
          [0.2, 0.3]]

# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts = np.zeros(1)
ypts = np.zeros(1)
labels = np.zeros(1)

## Zip : joins tuples together
## enumerate: add a enumerator to each eantry of the enumerate fucntion
for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    print(i, ((xmu, ymu), (xsigma, ysigma)))
    xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(200) * i))

# Visualize the test data
fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(xpts[labels == label], ypts[labels == label], '.')
ax0.set_title('Test data: 200 points.')
plt.show()

alldata = np.vstack((xpts, ypts))

#%%
'''
Documentation Source:
https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.html#cmeans

'''
#%%
# Set up the loop and plot
alldata = np.vstack((xpts, ypts))
fpcs = []


final_fuzzy_partitioning_coeff = []
for i in range(2,15):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, i, 2, error=0.001, maxiter=1000, init=None)
    final_fuzzy_partitioning_coeff.append(fpc)

plt.plot(final_fuzzy_partitioning_coeff)

