#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

from scipy.cluster.hierarchy import dendrogram, linkage

# On charge le dataset iris : 
iris = datasets.load_iris()

# On extrait X : 
X = iris.data

# On peut le transformer en DataFrame : 
X = pd.DataFrame(X)

# Cela permet d'appliquer la méthode .head : 
X.head()
Z = linkage(X, method="ward")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

_ = dendrogram(Z, p=10, truncate_mode="lastp", ax=ax)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.ylabel("Distance.")
plt.show()


# In[ ]:




