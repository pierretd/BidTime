#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


# In[2]:


n_rows = 300000
df = pd.read_csv("train.gz", nrows=n_rows, compression="infer")


# In[3]:


for c in df.columns:
    df[c]=df[c].apply(str)
    le=preprocessing.LabelEncoder().fit(df[c])
    df[c] =le.transform(df[c])
    pd.to_numeric(df[c]).astype(np.float)


# In[4]:


X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values


# In[5]:


n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]


# In[6]:


enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)


# In[7]:


n_features = int(X_train_enc.toarray().shape[1])
learning_rate = 0.001
n_iter = 50


# In[8]:


# Input and Target placeholders
x = tf.placeholder(tf.float32, shape=[None, n_features])
y = tf.placeholder(tf.float32, shape=[None])


# In[9]:


# Build the logistic regression model
W = tf.Variable(tf.zeros([n_features, 1]))
b = tf.Variable(tf.zeros([1]))


# In[10]:


logits = tf.add(tf.matmul(x, W), b)[:, 0]
pred = tf.nn.sigmoid(logits)


# In[11]:


cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
auc = tf.metrics.auc(tf.cast(y, tf.int64), pred)[1]


# In[12]:


optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[13]:


# Initialize the variables
init_vars = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


# In[14]:


batch_size = 1000


# In[15]:


indices = list(range(n_train))


# In[16]:


def gen_batch(indices):
    np.random.shuffle(indices)
    for batch_i in range(int(n_train / batch_size)):
        batch_index = indices[batch_i*batch_size: (batch_i+1)*batch_size]
        yield X_train_enc[batch_index], Y_train[batch_index]


# In[17]:


sess = tf.Session()

sess.run(init_vars)


# In[18]:


for i in range(1, n_iter+1):
    avg_cost = 0.
    for X_batch, Y_batch in gen_batch(indices):
        _, c = sess.run([optimizer, cost], feed_dict={x: X_batch.toarray(), y: Y_batch})
        avg_cost += c / int(n_train / batch_size)
    print('Iteration %i, training loss: %f' % (i, avg_cost))


# In[19]:


auc_test = sess.run(auc, feed_dict={x: X_test_enc.toarray(), y: Y_test})
print("AUC of ROC on testing set:", auc_test)


# In[ ]:




