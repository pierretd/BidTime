#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


# In[2]:


n_rows = 300000
df = pd.read_csv("train.gz", nrows=n_rows, compression="infer")


# In[3]:


df.shape


# In[4]:


for c in df.columns:
    df[c]=df[c].apply(str)
    le=preprocessing.LabelEncoder().fit(df[c])
    df[c] =le.transform(df[c])
    pd.to_numeric(df[c]).astype(np.float)


# In[5]:


Y = df['click'].values


# In[6]:


X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'],  
                axis=1).values


# In[7]:


print(X.shape)


# In[8]:


n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]


# In[9]:


enc = OneHotEncoder(handle_unknown='ignore')


# In[10]:


X_train_enc = enc.fit_transform(X_train)
print(X_train_enc[0])


# In[11]:


parameters = {'max_depth': [3, 10, None]}


# In[14]:


decision_tree = DecisionTreeClassifier(criterion='gini',
                                          min_samples_split=30)


# In[15]:


grid_search = GridSearchCV(decision_tree, parameters,
                              n_jobs=-1, cv=3, scoring='roc_auc')


# In[19]:


grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)


# In[20]:


decision_tree_best = grid_search.best_estimator_
pos_prob = decision_tree_best.predict_proba(X_test)[:, 1]
print('The ROC AUC on testing set is:{0:.3f}'.format(roc_auc_score(Y_test, pos_prob)))

