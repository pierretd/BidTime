#!/usr/bin/env python
# coding: utf-8

# # Avazu CTR Boosting Models

# 1) Import, clean, engineer features, hashing
# 2) Intro to boosting
# 3) Run a baseline model-no custom parameters
# 4) Add custom hyperparameters to model to reduce overfitting
# 5) Run CV to improve results and find best params
# 6) Run model with the improved params we have found

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import sklearn
import matplotlib.dates as mdates
import random
import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.preprocessing import OneHotEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Ignore Warnings

# In[2]:


'''Ignore Warning Messages'''
import warnings
warnings.filterwarnings('ignore')


# ### Matplotlib Style

# In[3]:


'''Set style to GGplot for asthetics'''
matplotlib.style.use('ggplot')


# ### Sample and Import Dataset

# In[4]:


''' Sample from Training Data down to 100,000 records'''

n = 40428966  #total number of records in the dataset 
sample_size = 100000
skip_values = sorted(random.sample(range(1,n), n-sample_size)) 

#Tracking the indices of rows to be skipped at random


# In[6]:


df = pd.read_csv('train.gz', compression='gzip', skiprows = skip_values)


# ## Data Prepation

# ### Date Time conversion with Hour feature

# In[139]:


def datesplit(originalDate):
    originalDate = str(originalDate)
    
    year = int("20" + originalDate[0:2])
    month = int(originalDate[2:4])
    day = int(originalDate[4:6])
    hour = int(originalDate[6:8])
    
    return datetime.datetime(year, month, day, hour)


# ### Create weekday and datetime from hour

# In[140]:


df['weekday'] = df['hour'].apply(lambda x : datesplit(x).weekday()) 


# In[141]:


df['hour'] = df['hour'].apply(lambda x : datesplit(x).hour)


# ### Variable Encoding

# In[142]:


model_features = ['hour', 'weekday', 'C1', 'banner_pos', 'site_category', 'app_category', 
                'device_type', 'device_conn_type', 'C15', 'C16', 'C18', 'C21']


# In[143]:


model_target = 'click'


# In[144]:


train_model = df[model_features+[model_target]]


# In[145]:


train_model = df[model_features+[model_target]]


# In[146]:


def one_hot_features(data_frame, feature_set):
    new_data_frame = pd.get_dummies(data_frame,
                                     columns = feature_set,
                                    sparse = True)

    return new_data_frame


# In[147]:


train_model = one_hot_features(train_model, ['hour', 'weekday', 'C1', 'banner_pos', 'site_category', 'app_category', 'device_type', 'device_conn_type', 'C15', 'C16', 'C18', 'C21'])


# In[148]:


model_features = np.array(train_model.columns[train_model.columns!=model_target].tolist())


# In[149]:


X = train_model[model_features].values


# In[150]:


y = train_model[model_target].values


# ### Test Train Splt

# In[151]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1,
                                                 random_state=100)


# ### Add site and user

# In[152]:


values = df['device_id'].value_counts().idxmax() 

df['user'] = np.where(df['device_id'] == values, df['device_ip'] + df['device_model'], df['device_id']) 


# In[153]:


df = df.drop(['device_id', 'device_model', 'device_ip'], axis=1) 


# In[154]:


df['site'] = df['site_id'] + df['site_domain'] 


# In[155]:


df = df.drop(['site_id', 'site_domain'], axis=1)


# # Hashing

# ### Create test train split

# In[81]:


X = df.drop(['click'], axis=1)
y = df['click']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=100)


# ### Hashing Function

# In[82]:


from sklearn.base import BaseEstimator, TransformerMixin

class MergeRareTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, col_names, threshold):
        self.col_names = col_names
        self.threshold = threshold

    def fit(self, X, y=None):

        X = pd.DataFrame(X)
        counts_dict_list = []
        
        for i in range(len(self.col_names)):
            serie = X[self.col_names[i]].value_counts()  # Series of each column
            rare_indexes = serie[serie<self.threshold[i]].index  # The indexes for least frequent features
            frequent_indexes = serie[serie>=self.threshold[i]].index  # The indexes for most frequent features
            dictionary = {x:'isRare' for x in rare_indexes}
            dictionary.update({x: str(x) for x in frequent_indexes})
            counts_dict_list.append(dictionary)   # Index for rare and most frequent
        self.counts_dict_list_ = counts_dict_list
        return self

    def transform(self, X):

        Xt = pd.DataFrame()
                
        for col, count_dict in zip(self.col_names, self.counts_dict_list_):
            Xt[col] = X[col].apply(lambda x:count_dict[x] if x in count_dict else 'isRare')
        # Represent the new features never observed and apply the count_dict to ther rest

        return Xt


# In[83]:


merged = MergeRareTransformer(col_names=Xtrain.columns, threshold=[20]*len(Xtrain.columns))
Xtrain_merged = merged.fit_transform(Xtrain)
Xtest_merged = merged.transform(Xtest)


# ### Test Hash

# In[84]:


import hashlib
hash('test hash') 


# ## Hashing

# ### Hash the Dataframe

# In[85]:


Xtrain_hashed = pd.DataFrame()
Xtest_hashed = pd.DataFrame()
for col in Xtrain_merged.columns:
    Xtrain_hashed[col] = Xtrain_merged[col].apply(lambda x : hash(str(x)) % 1000000) 
    Xtest_hashed[col] = Xtest_merged[col].apply(lambda x : hash( str(x)) % 1000000)


# In[86]:


import pandas as pd
import numpy as np
from IPython.display import display, HTML

CSS = """
.output {
    flex-direction: row;
}
"""

HTML('<style>{}</style>'.format(CSS))


# ### Log of hashed vs original feature dimentionality

# In[87]:


display(Xtrain.nunique()) 

display(Xtrain_hashed.nunique())


# # Introduction to XGBoost

# #### What is XGboost?
# 
# ##### Extreme Gradient Boosting
# 
# Focused on **computational speed** and **model performance**: few frills, but a good amount of advanced features
# 
# ##### Background Information
# 
# 
# 1) Original Creator was doing research on variants of tree boosting. He was looking to combine boosted trees with a conditional random field and found no existing model that satisfied his needs. 
# 
# 2) Author states that you cannot be successful with XGboost alone, you need to process data, and feature engineer first. 
# 
# ##### Model Features
# 
# 3 forms of gradient boosting are supported:
# 
# 1) **Gradient Boosting**- also called gradient boosting machine(w/ variable learning rate)
# 
# 2) **Stochastic Gradient Boosting**- with subsampling at the row, column, and column per split levels
# 
# 3) **Regularized Gradient Boosting**-with L1 and L2 regularization
# 
# ##### System Features
# 
# 1) **Parallelization** of tree-use all CPY cores during training
# 
# 2) **Distributed Computing** for training large models on a cluster of machines
# 
# 3) **Out of Core Computing** for large datasets that don't fit into memory
# 
# 4) **Cache Optimization** of data structures and algorithms to make best use of hardware
# 
# ##### Algorithm Features
# 
# 1) **Sparse Aware** implementation with automatic handling of missing data values
# 
# 2) **Block Structure** to support the parallelization of tree construction
# 
# 3) **Continued Training** so that you can further boost an already fitted model on new data
# 
# #### Why use XGB
# 
# **Execution Speed** and **Model performance**
# 
# Very Fast, has been benchmarked against other models and performs well against the feild 
# 
# ##### Model performance
# 
# XGB thrives with **structured or tabular datasets** on **classification** and **regression** predictive modeling problems
# 
# ##### What algorithm does it use?
# 
# **Gradient Boosted Decision Tree**: boosting is an ensemble technique where new models are added to correct errors made by existing models. Models are added sequentially until no further improvements can be made
# 
# **Gradient Boosting** is an approach where new models are created that predict the residuals or errors of prior models then adds them back together to make a final predictions.
# 
# It’s called **gradient boosting** because it uses a gradient descent algorithm to minimize the loss when adding new models
# 
# 

# ## Modeling

# ### XGboost with no parameters

# In[179]:


xgb = XGBClassifier(random_state=42)
xgb.fit(Xtrain_hashed, ytrain) 
soft = xgb.predict_proba(Xtest_hashed)
print("The ROC AUC is : " + str(roc_auc_score(ytest, soft[:,1])))


# Without any customization, XGboost scored a 70% ROC AUC. Next we will customize the classifier manaually to improve results.

# # Confusion Matrix

# In[180]:


y_pred = xgb.predict(Xtest_hashed)
predictions = [round(value) for value in y_pred]
print(confusion_matrix(ytest, predictions))


# # Classification Report

# In[207]:


fpr, tpr, thresholds = roc_curve(ytest, soft[:,1]) 


# In[208]:


xgboost_report = print(str(classification_report(ytest, predictions)))


# In[183]:


plt.figure(figsize=(12,6))
plt.plot(fpr, tpr, color='red', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGboost ROC Curve')
plt.show()


# ### XGBoost Parameters:
#     
#     

# ### XGboost with custom parameters

# In[210]:


xgb =XGBClassifier( 
    n_estimators=1000, 
    early_stopping_rounds=10,
    learning_rate=0.3, 
    n_jobs=-1, 
    random_state=42)
    
                             

xgb.fit(Xtrain_hashed, ytrain) 
soft = xgb.predict_proba(Xtest_hashed)
print("The ROC AUC is : " + str(roc_auc_score(ytest, soft[:,1])))


# By customizing out Classifier, we gained 2 points of ROC AUC with our model. 

# In[211]:


print(str(classification_report(ytest, predictions)))


# In[212]:


fpr, tpr, thresholds = roc_curve(ytest, soft[:,1]) 


# In[214]:


plt.figure(figsize=(12,6))
plt.plot(fpr1, tpr1, color='red', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGboost ROC Curve')
plt.show()


# ## Cross Validation with multiple parameters

# XGboost allows us to declare multiple values for a parameter, and with a cross validation, we can further improve our model by increasing the amount of possible model combinations. This can also decrease a model's perfomance, if there is too much overfitting due to the cross validation.

# In[191]:


xgb_model = XGBClassifier(    n_estimators=1000, 
                              early_stopping_rounds=10,
                              n_jobs=-1, 
                              random_state=42)

parameters = {'learning_rate': [.25, .3, .35],
              }

gs_xgb = GridSearchCV(xgb_model, parameters, cv=3, 
                      scoring='roc_auc', verbose=False)
gs_xgb.fit(Xtrain_hashed, ytrain)
gs_xgb.best_score_
soft = gs_xgb.predict_proba(Xtest_hashed)
print("The ROC AUC is : " + str(roc_auc_score(ytest, soft[:,1])))


# The model improved, but only by .02. The more tuning you do to your hyperparameters the smaller the increases will become. This is why feature engineering and proper data cleaning are paramount to a successful model. Tuning your model will help results, but it won't create a good model from bad data.

# In[215]:


print(str(classification_report(ytest, predictions)))


# In[216]:


xgb_model = XGBClassifier(    
                              early_stopping_rounds=10,
                              learning_rate=0.3, 
                              n_jobs=-1, 
                              random_state=42)

parameters = {'min_child_weight ': [1, 5, 10],
              }

gs_xgb = GridSearchCV(xgb_model, parameters, cv=3, 
                      scoring='roc_auc', verbose=False)
gs_xgb.fit(Xtrain_hashed, ytrain)
gs_xgb.best_score_
soft = gs_xgb.predict_proba(Xtest_hashed)
print("The ROC AUC is : " + str(roc_auc_score(ytest, soft[:,1])))


# In[ ]:


The changing the n_estimators gave us even less improvement. 


# # Model Interpretability and PDP plots

# In[132]:


xgb.fit(Xtrain_hashed, ytrain)


# In[133]:


importance = pd.Series(gb.feature_importances_, Xtrain_hashed.columns)


# In[134]:


importance.sort_values(ascending=False) 


# We can see what features are most and least important to our model, and create a new model with different features by examing the features of importance

# ### Cross validation-best cv with multiple parameters

# Above we ran two models where we changed min_child_weight and learning_rate independtenly. Let's changing them together to see what the best combinations of paramaters might be.

# In[217]:


child_rate_grid = {'min_child_weight': [1, 5, 10], 'learning_rate': [.25, .3, .35]}

tuning = GridSearchCV(estimator =XGBClassifier(
    n_estimators=1000,
    monotone_constraints='(0)',
    random_state=42), 
    param_grid = child_rate_grid,
    scoring='roc_auc',
    iid=False,
    cv=3)
tuning.fit(X_train,y_train)
tuning.grid_scores_, tuning.best_params_, tuning.best_score_


# Tuning also supports multiple values for multiple parameters at once. We see here that the min child weight doesn't seem to change results too much, but max depth of 3 shows much better results than max depth of 25.

# ## K Folds

# A KFold is another type of cross validation that allows you split data into k consecutive folds and one fold is used finding error, while the rest train the system. This proccess allows your model to be robust to different shards of data. 

# In[233]:


kfold = 3
skf = StratifiedKFold(n_splits=kfold, random_state=42)


# In[234]:


# XGBoost parameters
params_overfit = {
    'n_estimators' : 1000,
    'eval_metric': 'auc',
    'min_child_weight': 5,
    'learning_rate' : .1,
    'iid': False,
    }


# In[235]:


models_by_fold = []


# In[236]:


#Kfold with overfitting

from sklearn.model_selection import StratifiedKFold 
import xgboost as xgb
# Kfold training
for train_index, test_index in skf.split(X, y):
    
    # Convert data into XGBoost format.
    d_train = xgb.DMatrix(Xtrain_hashed, ytrain)
    d_valid = xgb.DMatrix(Xtest_hashed, ytest)
    
    # Watchlist to evaluate results while training.
    watchlist = [(d_train, 'train'), (d_valid, 'test')]

    # Training this fold
    mdl = xgb.train(params_overfit, d_train, 1000, watchlist, early_stopping_rounds=150, maximize=True, verbose_eval=100)
    
    # Add model to the list of models (one for each fold)
    models_by_fold.append(mdl)


# One potential isssue that can arise when using resampling is overfitting of your model. While our train AUC is high, our test AUC is laggin behind. This tells us that our model is not generazable.This can be avoided by customizing parameters that help reduce overfitting.

# In[237]:


# XGBoost parameters
params_adjusted = {
    'min_child_weight': 10.0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 5,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'num_boost_round' : 700
    }


# In[238]:


models_by_fold = []


# In[239]:


#Kfold adjusted for overfitting

from sklearn.model_selection import StratifiedKFold 
import xgboost as xgb
# Kfold training
for train_index, test_index in skf.split(X, y):
    
    # Convert data into XGBoost format.
    d_train = xgb.DMatrix(Xtrain_hashed, ytrain)
    d_valid = xgb.DMatrix(Xtest_hashed, ytest)
    
    # Watchlist to evaluate results while training.
    watchlist = [(d_train, 'train'), (d_valid, 'test')]

    # Training this fold
    mdl = xgb.train(params_adjusted, d_train, 1000, watchlist, early_stopping_rounds=150, maximize=True, verbose_eval=100)
    
    # Add model to the list of models (one for each fold)
    models_by_fold.append(mdl)


# The gap between train and test is almost half as small due to the new parameters intended to reduce overfitting. This leaves us with a more generalizable model that will be more robust to incoming data.
