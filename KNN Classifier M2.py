#!/usr/bin/env python
# coding: utf-8

# In[16]:


corpus = open('Movies_TV.txt').read()


# In[17]:


import re
corpus = re.sub(r'Domain.*\n', '', corpus)


# In[18]:


rows = corpus.split('\n')
rows.remove(rows[-1])

X, y = [], []
for row in rows:
    _, label, _, review = row.split('\t')
    X.append(review)
    y.append(label)

len(X), len(y)


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
matrix_X = tfidf.fit_transform(X)

X[0]


# In[20]:


from sklearn.neighbors import KNeighborsClassifier

knnc = KNeighborsClassifier(n_neighbors = 5)
knnc.fit(matrix_X[:-10], y[:-10])


# In[21]:


knnc.predict(matrix_X[-10:])


# In[22]:


y[-10:]

