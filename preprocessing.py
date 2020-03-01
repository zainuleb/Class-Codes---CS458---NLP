#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk import ngrams


# In[2]:


text = "i hope you are you are doing well"
words = text.split(' ')
words


# In[3]:


from string import punctuation as punc
punc


# In[4]:


from sklearn.feature_extraction import stop_words
list(stop_words.ENGLISH_STOP_WORDS)


# In[5]:


#words = words[:100]
unigrams = list(ngrams(words, 1))
bigrams = list(ngrams(words, 2))
trigrams = list(ngrams(words, 3))
print("unigrams", unigrams)
print("bigrams", bigrams)
print("trigrams", trigrams)


# In[6]:


from nltk.stem import PorterStemmer
ps = PorterStemmer()


# In[7]:


for word in words:
    print(ps.stem(word))


# In[8]:


from nltk.stem import WordNetLemmatizer
wl = WordNetLemmatizer()


# In[9]:


for word in words:
    print(wl.lemmatize(word, 'v'))


# In[10]:


from nltk import pos_tag
tags = pos_tag(words)
tags


# In[11]:


import nltk
nltk.download('averaged_perceptron_tagger')


# In[12]:


import nltk
nltk.download('wordnet')


# In[13]:


print('unigrams: ', unigrams)
print('bigrams: ', bigrams)
print('trigrams: ', trigrams)


# In[14]:


unigrams_freq = [words.count(x)/len(set(words)) for x in words]
unigrams_freq


# In[15]:


bigrams[0]


# In[16]:


bigrams_freq = []
for b in bigrams:
    temp = bigrams.count(b)
    temp2 = words.count(b[0])
    bigrams_freq.append(temp/temp2)
bigrams_freq    


# In[17]:


trigrams_freq = [trigrams.count(x)/bigrams.count(x[:2]) for x in trigrams]
trigrams_freq


# In[18]:


from nltk.corpus import brown


# In[19]:


words = brown.words(categories = 'news')


# In[20]:


len(words)

