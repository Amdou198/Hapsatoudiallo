#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans

# On charge le dataset iris : 
iris = datasets.load_iris()

# On extrait X : 
X = iris.data

# On peut le transformer en DataFrame : 
X = pd.DataFrame(X)

# Cela permet d'appliquer la méthode .head : 
X.head()

# Une liste vide pour enregistrer les inerties :  
intertia_list = [ ]

# Notre liste de nombres de clusters : 
k_list = range(1, 10)

# Pour chaque nombre de clusters : 
for k in k_list : 
    
    # On instancie un k-means pour k clusters
    kmeans = KMeans(n_clusters=k)
    
    # On entraine
    kmeans.fit(X)
    
    # On enregistre l'inertie obtenue : 
    intertia_list.append(kmeans.inertia_)
fig, ax = plt.subplots(1,1,figsize=(12,6))

ax.set_ylabel("intertia")
ax.set_xlabel("n_cluster")

ax = plt.plot(k_list, intertia_list)


# In[ ]:





# In[ ]:




