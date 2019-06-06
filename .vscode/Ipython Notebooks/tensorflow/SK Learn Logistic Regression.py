#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier


# In[30]:


n_rows = 300000
df = pd.read_csv("train.gz", nrows=n_rows, compression="infer")


# In[31]:


for c in df.columns:
    df[c]=df[c].apply(str)
    le=preprocessing.LabelEncoder().fit(df[c])
    df[c] =le.transform(df[c])
    pd.to_numeric(df[c]).astype(np.float)


# In[32]:


X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values


# In[33]:


n_train = 100000
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]


# In[34]:


enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)


# In[35]:


sgd_lr = SGDClassifier(loss='log', penalty=None, fit_intercept=True, max_iter=10, learning_rate='constant', eta0=0.01)
sgd_lr.fit(X_train_enc.toarray(), Y_train)

pred = sgd_lr.predict_proba(X_test_enc.toarray())[:, 1]
print('Training samples: {0}, AUC on testing set: {1:.3f}'.format(n_train, roc_auc_score(Y_test, pred)))

