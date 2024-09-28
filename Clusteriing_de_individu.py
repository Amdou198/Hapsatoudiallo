#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# On charge le dataset iris : 
iris = datasets.load_iris()

# On extrait X : 
X = iris.data

# On le transforme en DataFrame pour pouvoir mieux visualiser nos données : 
X = pd.DataFrame(X)
X.head()

# On instancie notre Kmeans avec 3 clusters : 
kmeans = KMeans(n_clusters=3)

# On l'entraine : 
kmeans.fit(X)

# On peut stocker nos clusters dans une variable labels : 
labels = kmeans.labels_
labels

# On peut stocker nos centroids dans une variable : 
centroids = kmeans.cluster_centers_
centroids

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# elaboration du PCA

# In[33]:


pca = PCA(n_components=4)
pca.fit(X_scaled)


# Calcul de variance cumulée

# In[36]:


pca.explained_variance_ratio_.cumsum()


# Visualisation avec la methode coude

# In[39]:


plt.plot(range(1,5), pca.explained_variance_ratio_.cumsum())
plt.xlabel = "n_components"
plt.ylabel = "% of variance"


# Calcul des cordonnées

# In[42]:


X_proj = pca.transform(X_scaled)
X_proj[:10]


# Pour plus de clarté

# In[45]:


X_proj = pd.DataFrame(X_proj, columns = ["F1", "F2", "F3", "F4"])
X_proj[:10]


# Calcul de centroide

# In[48]:


centroids_scaled = scaler.fit_transform(centroids)
centroids_scaled


# Un dataframme pour plus de clarté

# In[51]:


centroids_proj = pca.transform(centroids_scaled)
centroids_proj = pd.DataFrame(centroids_proj, 
                              columns = ["F1", "F2", "F3", "F4"], 
                              index=["cluster_0", "cluster_1", "cluster_2"])
centroids_proj


# Visualisation

# In[54]:


plt.scatter(X_proj.iloc[:, 0], X_proj.iloc[:, 1])


# Une visualisation encore mieux

# In[57]:


fig, ax = plt.subplots(1,1, figsize=(8,7))

ax.scatter(X_proj.iloc[:, 0], X_proj.iloc[:, 1], c= labels, cmap="Set1")

ax.set_xlabel("F1")
ax.set_ylabel("F2")


# Elaboration de centroide dans nos different clusters

# In[60]:


fig, ax = plt.subplots(1,1, figsize=(8,7))

ax.scatter(X_proj.iloc[:, 0], X_proj.iloc[:, 1], c= labels, cmap="Set1", alpha=0.3)
ax.scatter(centroids_proj.iloc[:, 0], centroids_proj.iloc[:, 1],  marker="s", c="black" )

ax.set_xlabel("F1")
ax.set_ylabel("F2")


# une visualisation à trois dimension

# In[63]:


fig= plt.figure(1, figsize=(8, 6))

ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
ax.scatter(
    X_proj.iloc[:, 0],
    X_proj.iloc[:, 1],
    X_proj.iloc[:, 2],
    c=labels,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40)

ax.set_xlabel("F1")
ax.set_ylabel("F2")
ax.set_zlabel("F3")


# Visualisation avec les axes d'inertie

# In[69]:


fig, ax = plt.subplots(1,1, figsize=(8,7))

#Transforme notre DataFrame d'origine
X_ = np.array(X_proj)

# On enregistre nos axes x, y
x, y = axis = (0,1 )

# plus besoin d'utiliser iloc
ax.scatter(X_[:, 0], X_[:, 1], c= labels, cmap="Set1")

# nom des axes, avec le pourcentage d'inertie expliqué
ax.set_xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
ax.set_ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

# affichage des lignes horizontales et verticales
plt.plot([-3, 3], [0, 0], color='grey', alpha=0.8)
plt.plot([0,0], [-3, 3], color='grey', alpha=0.8)

# on rajoute un tritre
plt.title("Projection des individus (sur F{} et F{})".format(x+1, y+1), )


# In[ ]:




