#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing and Manipulating Data

# Importing and cleaning data

# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[72]:


pop = pd.read_csv('pop_data.csv')
rap = pd.read_csv('rap_data.csv')
rnb = pd.read_csv('rnb_data.csv')

pop['genre'] = 'pop'
rap['genre'] = 'rap'
rnb['genre'] = 'rnb'

alldata = pd.concat([pop, rap, rnb], axis=0)
alldata = alldata.drop(alldata.columns[0], axis = 1)
alldata.head()


# ___________

# ## Visualization

# ### Danceability in Relation to Energy and Tempo

# We wanted to take a closer look at the "danceability" feature for tracks since this is a less objective and typical variable than others. According to the Spotify API Audio Features documentation, they described "danceability" as "how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity." Hence, we decided to compare "danceability" to "energy" and "tempo" and plot the data with matplotlib. We had to convert the categorical variable of "genre" to a numerical code so that we may use this to color the data. Each different color of a point represents a different genre. There was some noticable clumping by different genre for danceability but further statistical tests would need to be conducted to validate this conclusion.

# In[73]:


gen = alldata['genre']
genclr = gen.astype("category").cat.codes

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.scatter(alldata['danceability'], alldata['energy'], alldata['tempo'], c=genclr.values.ravel(), cmap = 'rainbow')

ax.set_title('Danceability in Realtion to Energy and Tempo')
ax.set_xlabel('Danceability')
ax.set_ylabel('Energy')
ax.set_zlabel('Tempo')

plt.show()


# ### Correlation Plot

# Since there were so many audio features, we also wanted to look into how correlated the different features were to each other. We found that the 2 most correlated features was "energy" and "loudness".

# In[74]:


corr = alldata.corr()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(corr)

ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns)
for tick in ax.get_xticklabels():
        tick.set_rotation(90)

ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)

plt.show()

