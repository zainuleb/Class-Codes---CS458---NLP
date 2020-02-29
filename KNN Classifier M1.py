#!/usr/bin/env python
# coding: utf-8

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


# In[23]:


#corpus = open('D:\\Datasets\grasshoppers_katydids.csv').read()
corpus = pd.read_csv('grasshoppers_katydids.csv')


# In[24]:


#rows = corpus.split('\n')
X = corpus.iloc[: , [1, 2]].values
y = corpus.iloc[: , 3].values

set(y)


# In[28]:


knnc = KNeighborsClassifier(n_neighbors = 3)
knnc.fit(X, y)
res = knnc.predict([[4, 5]])


# In[29]:


import matplotlib.pyplot as plt
plt.scatter(X[: , 0], X[: , 1], s = 10, c = 'black', marker = 'o')

labels = knnc.predict(X)
plt.scatter(X[labels=='Grasshopper', 0], X[labels=='Grasshopper', 1], s = 50, marker = 'o', color = 'blue')
plt.scatter(X[labels=='Katydid', 0], X[labels=='Katydid', 1], s = 50, c = 'red', marker = 's')
plt.scatter(4, 5, s = 100, marker = 'D', color = 'purple')
plt.title('Grasshopper and Katydids Classification')
plt.xlabel('Abdomen length')
plt.ylabel('Antennae length')
plt.show()

