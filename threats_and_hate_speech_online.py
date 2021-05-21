#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import spacy
import csv
from tqdm import tqdm
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from spacytextblob.spacytextblob import SpacyTextBlob
spacy_text_blob = SpacyTextBlob()
# Initialize spacy by creating a standard spacy object
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(spacy_text_blob)


# In[3]:


threats_path = os.path.join("..", "data", "VideoCommentsThreatCorpus.txt")


# In[4]:


# create comments list
comments = [] 
# open the txt file
with open(threats_path, "r", encoding="utf-8") as file: 
    #for every line in the txt file
    for line in file: 
        # if that line starts with a 0 or a 1...
        if line[0]=="0" or line[0]=="1":
            # ... append it to the comments list
            comments.append((line[0], line[1:].rstrip())) # use r.strip() to remove newlines at the end
        # otherwise, just ignore that line
        else:
            pass
# Create datafame from list of tuples, one column with number, one with text of comment
data = pd.DataFrame(comments, columns=["code", "text"])


# In[6]:


data


# In[62]:


# what is the distribution of threats and non-threats? where 1 is a threat and 0 is not
data["code"].value_counts()


# In[64]:


adj_count = 0

# process text in batches 
# this wil create a doc which can be iterated over.
# so everytime it goes through a batch it will make a doc
for doc in nlp.pipe(data["text"], batch_size=500):
    # go through
    for token in doc:
        # if the token is an adjective
        if token.tag_ == "JJ":
            # add it to the count
            adj_count += 1


# In[65]:


# number of adjectives in the text
print(adj_count)


# In[66]:





# In[ ]:




