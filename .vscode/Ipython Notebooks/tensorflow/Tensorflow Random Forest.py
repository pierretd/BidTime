#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
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

print(X.shape, Y.shape)


# In[5]:


df.head()


# In[6]:


n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]


# In[7]:


enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)


# In[8]:


get_ipython().system('pip install tensorflow')
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources


# In[9]:


n_iter = 50
n_classes = 2
n_features = int(X_train_enc.toarray().shape[1])
n_trees = 10
max_nodes = 30000


# In[10]:


x = tf.placeholder(tf.float32, shape=[None, n_features])
y = tf.placeholder(tf.int64, shape=[None])


# In[11]:


hparams = tensor_forest.ForestHParams(num_classes=n_classes, num_features=n_features, num_trees=n_trees,
                                      max_nodes=max_nodes, split_after_samples=30).fill()


# In[12]:


forest_graph = tensor_forest.RandomForestGraphs(hparams)


# In[13]:


train_op = forest_graph.training_graph(x, y)
loss_op = forest_graph.training_loss(x, y)


# In[14]:


infer_op, _, _ = forest_graph.inference_graph(x)


# In[15]:


auc = tf.metrics.auc(tf.cast(y, tf.int64), infer_op[:, 1])[1]


# In[16]:


init_vars = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

sess = tf.Session()

sess.run(init_vars)


# In[17]:


batch_size = 1000


# In[18]:


indices = list(range(n_train))


# In[19]:


def gen_batch(indices):
    np.random.shuffle(indices)
    for batch_i in range(int(n_train / batch_size)):
        batch_index = indices[batch_i*batch_size: (batch_i+1)*batch_size]
        yield X_train_enc[batch_index], Y_train[batch_index]


# In[20]:


for i in range(1, n_iter + 1):
    for X_batch, Y_batch in gen_batch(indices):
        _, l = sess.run([train_op, loss_op], feed_dict={x: X_batch.toarray(), y: Y_batch})
    acc_train = sess.run(auc, feed_dict={x: X_train_enc.toarray(), y: Y_train})
    print('Iteration %i, AUC of ROC on training set: %f' % (i, acc_train))
    acc_test = sess.run(auc, feed_dict={x: X_test_enc.toarray(), y: Y_test})
    print("AUC of ROC on testing set:", acc_test)


# In[21]:


#With no feature engineering, we achieved a ROCAUC of almost 80

