#!/usr/bin/env python
# coding: utf-8

# # Predicting Genre of Songs based on Spotify API Data Features

# Nathan Zheng, Christopher Yuan, Kassie Wang

# ## Problem

# The problem our group is trying to tackle is to predict the genre of song based on the Spotify API Data Features. The Spotify API provides a wide range of data features that the company analyzed from each audio track, most of which is numerical data. Different features include acousticness, danceability, energy, instrumentalness, key, liveness, loudness, tempo, time signature, and duration. These features are documented here: https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/ 
# 
# Because there isn’t a Spotify dataset online to satisfy our needs, we intend to create our own dataset using the API. Our main objective of this project is to discern whether we can use machine learning models to classify songs accurately within several main genres, using a small sample size of notable artists from each genre to generate our data. We will be choosing three artists for each of three different music genres: mainstream pop, rap, and R&B. For pop, we chose Ed Sheeran, Selena Gomez, and Justin Bieber. For rap, we chose Drake, Eminem, and Kendrick Lamar. Lastly, for R&B, we chose Daniel Caesar, H.E.R, and Kehlani.
# 	

# ## Hypothesis

# With this data, we will use a KNN and Decision Tree Classifier to test our results. We hypothesize that both of these models should be able to decently classify our data, though we are not sure which one will be more accurate. 

# ## Formatting the Data

# We created a program to scrape data from the Spotify API and format them in csv files based on genre. To run the data through our models, we simply added a label for genre and combined the datasets into one. 
# 
# The program we used to collect the data is included in the zip file under 'collect_data.py'.

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import datasets
from sklearn import mixture


# In[7]:


pop = pd.read_csv('pop_data.csv')
rap = pd.read_csv('rap_data.csv')
rnb = pd.read_csv('rnb_data.csv')


# In[8]:


pop['genre'] = 'pop'
rap['genre'] = 'rap'
rnb['genre'] = 'rnb'

alldata = pd.concat([pop, rap, rnb], axis=0)
alldata = alldata.drop(alldata.columns[0], axis = 1)
alldata.head()


# ## Visualization

# ### Danceability in Relation to Energy and Tempo

# We wanted to take a closer look at the "danceability" feature for tracks since this is a less objective and typical variable than others. According to the Spotify API Audio Features documentation, they described "danceability" as "how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity." Hence, we decided to compare "danceability" to "energy" and "tempo" and plot the data with matplotlib. We had to convert the categorical variable of "genre" to a numerical code so that we may use this to color the data. Each different color of a point represents a different genre. There was some noticable clumping by different genre for danceability but further statistical tests would need to be conducted to validate this conclusion.

# In[24]:


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

# In[25]:


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


# ## Models

# ### K-Nearest Neighbors

# The first model we used to classify the dataset was a KNN. KNN was an appropriate model for our data because it would be able to split the data into more than two categories like we wanted. Also, from the first visualization we could see some clumping in our dataset and we theorized that KNN would able to take advantage of these clumps to give an accurate prediction. For KNN model, we had our algorithm take in all features that had a numerical value. 

# In[26]:


X=alldata.drop(['analysis_url','id','track_href','type','uri','genre'],axis=1)
y=alldata['genre']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("KNN Accuracy Score: ", accuracy_score(y_test, predictions))


# ### Decision Tree Classifier

# The second model we used was the decision tree classifier. Like the KNN, the decision tree is able to classify data into multiple categories. We compared the accuracy score of the KNN and the decision tree classifier. 

# In[40]:


model=tree.DecisionTreeClassifier(max_depth=10)

model.fit(x_train,y_train)

dtree_pred_train = model.predict(x_train)
dtree_pred_test = model.predict(x_test)

print("Decision Tree Accuracy Score: ", accuracy_score(y_test, dtree_pred_test))


# ## Results/Conclusion

# Looking at the accuracy scores of the two models we used, the Decision Tree Classifier was a better model for the data. Both models were able to classify the data to an extent: randomly choosing songs and placing them in categories should yield a score of 0.33, and both models achieved a score higher than that. Our hypothesize was pretty much proved correct, and both models do classify the data relatively accurately. 
# 
# In the future, some optimizations we could use are to decrease the number of features used to ones that are more relevant, or use other models to classify the data such as ones that use unsupervised learning. 

# In[ ]:




