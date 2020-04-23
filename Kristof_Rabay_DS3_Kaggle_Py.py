#!/usr/bin/env python
# coding: utf-8

# ### Why I used Python (not only R) for the Kaggle competition

# After using R to stack a GBM and an RF together with a neural net metalearner in h2o, and trying out Keras and xgBoost separately, I wanted to enhance my stacking model with xgBoost (Windows does not let me in h2o). Meanwhile, I also wanted to see why so many Kagglers use LightGBM for their submissions. After numerous attempts and fails to install LightGBM to R, I decided to turn to Python, re-do the whole assignment from scratch and build an ensemble in skLearn with GBM, LightGBM, xgBoost, RF and possibly a CatBoost, all as base-learners

# In[2]:


import pandas as pd
import numpy as np


# In[5]:


train = pd.read_csv("C://Users/Krisz/Desktop/ceu/materials/winter_1/ml/own/ds3_assignments/kaggle_ds3/train.csv")
test = pd.read_csv("C://Users/Krisz/Desktop/ceu/materials/winter_1/ml/own/ds3_assignments/kaggle_ds3/test.csv")


# I know from the time spent in R which features to drop and which to scale...

# In[6]:


train.drop(['url', ' timedelta', ' n_non_stop_words', ' n_non_stop_unique_tokens', ' kw_avg_min'], axis=1, inplace=True)
test.drop(['url', ' timedelta', ' n_non_stop_words', ' n_non_stop_unique_tokens', ' kw_avg_min'], axis=1, inplace=True)


# In[ ]:


train.rename(columns = lambda x: x.strip(), inplace = True)
test.rename(columns = lambda x: x.strip(), inplace = True)


# In[7]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
scaler = StandardScaler()


# In[8]:


cols_to_scale = ['n_tokens_title' , 'n_tokens_content' , 'n_unique_tokens' , 'n_non_stop_words' , 'n_non_stop_unique_tokens' , 'num_hrefs' , 'num_self_hrefs' , 'num_imgs' , 'num_videos' , 'average_token_length' , 'num_keywords' , 'kw_min_min' , 'kw_max_min' , 'kw_avg_min' , 'kw_min_max' , 'kw_max_max' , 'kw_avg_max' , 'kw_min_avg' , 'kw_max_avg' , 'kw_avg_avg' , 'self_reference_min_shares' , 'self_reference_max_shares' , 'self_reference_avg_sharess' , 'LDA_00' , 'LDA_01' , 'LDA_02' , 'LDA_03' , 'LDA_04' , 'global_subjectivity' , 'global_sentiment_polarity' , 'global_rate_positive_words' , 'global_rate_negative_words' , 'rate_positive_words' , 'rate_negative_words' , 'avg_positive_polarity' , 'min_positive_polarity' , 'max_positive_polarity' , 'avg_negative_polarity' , 'min_negative_polarity' , 'max_negative_polarity' , 'title_subjectivity' , 'title_sentiment_polarity' , 'abs_title_subjectivity' , 'abs_title_sentiment_polarity']


# In[9]:


train[train.columns.intersection(cols_to_scale)] = scaler.fit_transform(train[train.columns.intersection(cols_to_scale)])
test[test.columns.intersection(cols_to_scale)] = scaler.transform(test[test.columns.intersection(cols_to_scale)])


# In[10]:


y_train = train['is_popular']
X_train = train.drop(['is_popular'], axis=1)
X_test = test # no need for .copy() as the two will remain identical, it's just a naming convention


# Splitting the train set to an actual train and a validation (serving as test), and the original test will be submitted to Kaggle

# In[25]:


X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 20202020)


# Everything below is already tuned, so the 'grid search' isn't actually a grid, it's the final, best parameters

# ## LightGBM - awesome tool, easy to use, very powerful even for the first, un-tuned try

# In[26]:


import lightgbm as lgbm


# In[27]:


estimator = lgbm.LGBMClassifier(boosting_type= 'dart', 
                                metric = 'auc', 
                                objective='binary', 
                                application = 'binary',
                                n_estimators = 500, 
                                random_state = 20202020)

param_grid = {
    'learning_rate': [0.05],
    'num_leaves' : [60],
    'min_data_in_leaf': [20],   
    'feature_fraction': [2/3],     
    'drop_rate': [0.15]}

scoring = {'AUC': 'roc_auc'}

lgbm_grid = GridSearchCV(estimator, param_grid, cv = 3, scoring = scoring, refit = 'AUC')

lgbm_grid.fit(X_tr, y_tr)


# In[28]:


lgbm_grid.best_score_


# In[29]:


lgbm_grid.best_params_


# In[30]:


results = lgbm_grid.best_estimator_.predict_proba(X_val)[:,1]
act = y_val.array

roc_auc_score(act, results)


# #### AUC: 70.96 best on validation

# ## GBM

# In[31]:


from sklearn.ensemble import GradientBoostingClassifier


# In[32]:


param_grid = {'learning_rate':[0.01], 
              'n_estimators': [500], 
              'max_features' : [0.6], 
              'subsample' : [0.3], 
              'min_samples_split' : [2],
              'max_depth' : [9], 
              'min_samples_leaf' : [10]}

gbm = GradientBoostingClassifier(random_state = 20202020)

grid_search_gbm = GridSearchCV(estimator = gbm,
                               param_grid = param_grid, 
                               scoring = 'roc_auc',
                               n_jobs = -1,
                               cv=3, 
                               verbose = 0)

grid_search_gbm.fit(X_tr, y_tr)


# In[33]:


grid_search_gbm.best_score_


# In[34]:


grid_search_gbm.best_params_


# In[35]:


results = grid_search_gbm.best_estimator_.predict_proba(X_val)[:,1]
act = y_val.array

roc_auc_score(act, results)


# #### AUC: 71.037 best on validation

# ## xgBoost

# In[36]:


from xgboost import XGBClassifier


# In[37]:


params = {'min_child_weight': [3],
          'gamma': [0],
          'subsample': [0.75],
          'colsample_bytree': [0.45],
          'max_depth': [7], 
          'eta' : [0.01]}

xgb = XGBClassifier(n_estimators = 500, 
                    objective='binary:logistic',
                    verbosity = 0, 
                    random_state = 20202020, 
                    eval_metric = 'auc')

grid_search_xgb = GridSearchCV(estimator = xgb,
                               param_grid = params, 
                               scoring = 'roc_auc',
                               n_jobs = -1,
                               cv = 3, 
                               verbose = 0)

grid_search_xgb.fit(X_tr, y_tr)


# In[38]:


grid_search_xgb.best_score_


# In[39]:


grid_search_xgb.best_params_


# In[40]:


results = grid_search_xgb.best_estimator_.predict_proba(X_val)[:,1]
act = y_val.array

roc_auc_score(act, results)


# #### AUC: 71.329 best on validation

# ## Random Forest

# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


param_grid = {
    'max_depth': [20],
    'max_features': [8],
    'min_samples_leaf': [11],
    'min_samples_split': [6],
    'n_estimators': [500]
}

rf = RandomForestClassifier(random_state = 20202020)

grid_search_rf = GridSearchCV(estimator = rf, 
                              param_grid = param_grid, 
                              cv = 3, 
                              n_jobs = -1, 
                              verbose = 0, 
                              scoring = 'roc_auc')

grid_search_rf.fit(X_tr, y_tr)


# In[43]:


grid_search_rf.best_score_


# In[44]:


grid_search_rf.best_params_


# In[45]:


results = grid_search_rf.best_estimator_.predict_proba(X_val)[:,1]
act = y_val.array

roc_auc_score(act, results)


# #### AUC: 70.926 best on validation

# ## VotingClassifier

# Found this ensemble method in skLearn, turned out to be better (validation AUC) than stacking.

# In[46]:


from sklearn.ensemble import VotingClassifier


# In[47]:


estimators = [('gbm', grid_search_gbm.best_estimator_),
              ('xgb', grid_search_xgb.best_estimator_),
              ('lgbm', lgbm_grid.best_estimator_)]
              #('rf', grid_search_rf.best_estimator_)] # validation AUC was better without Random Forest model


# In[48]:


vclf = VotingClassifier(estimators = estimators, 
                        voting = 'soft')

vclf.fit(X_tr, y_tr)


# To my understanding the 'soft' voting is better here because my base learners are well tuned. This way the VotingClassifier takes the average of the probabilities. Hard voting would use a majority vote and return 1 and 0 classes instead of probabilities

# In[49]:


results = vclf.predict_proba(X_val)[:,1]
act = y_val.array

roc_auc_score(act, results)


# #### AUC: 71.339 best on validation

# ### To submit 

# In[51]:


submission = pd.DataFrame({ 'article_id': test['article_id'],
                            'score': vclf.predict_proba(X_test)[:,1]})


# In[52]:


submission.to_csv("Kr_Rab_Py.csv", index=False)


#  

# Below are algos I didn't end up using at all: StackingClassifier and CatBoost

#  

# ## StackingClassifier - won't submit, VotingClassifier was better

# In[1]:


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


# In[256]:


estimators = [('gbm', grid_search_gbm.best_estimator_),
              #('xgb', grid_search_xgb.best_estimator_),
              ('lgbm', lgbm_grid.best_estimator_)]
              #('rf', grid_search_rf.best_estimator_)]


# In[257]:


clf = StackingClassifier(estimators = estimators, 
                         final_estimator = LogisticRegression(random_state = 20202020), # logreg is better than gbm
                         stack_method = 'predict_proba')

clf.fit(X_tr, y_tr)


# In[258]:


results = clf.predict_proba(X_val)[:,1]
act = y_val.array

roc_auc_score(act, results)


# #### 71.246 best on validation
# with lgbm and gbm as base learners

# ## CatBoost (left out of Stack model - takes forever to train)

# In[111]:


from catboost import CatBoostClassifier


# In[116]:


params = {'border_count': [254],
          'l2_leaf_reg': [0],
          'depth': [12],
          'metric_period': [75],
          'bagging_temperature': [0.2], 
          'learning_rate' : [0.02], 
          'min_data_in_leaf' : [10]}

cbc = CatBoostClassifier(iterations = 500, 
                         random_seed = 20202020, 
                         eval_metric = 'AUC')

grid_search_cbc = GridSearchCV(estimator = cbc,
                               param_grid = params, 
                               scoring = 'roc_auc',
                               n_jobs = -1,
                               cv = 3, 
                               verbose = 0)

grid_search_cbc.fit(X_tr, y_tr)


# In[117]:


grid_search_cbc.best_score_


# In[118]:


grid_search_cbc.best_params_


# In[119]:


results = grid_search_cbc.best_estimator_.predict_proba(X_val)[:,1]
act = y_val.array

roc_auc_score(act, results)


# #### CatBoost takes forever and AUC would need huge improvement, so I'm dropping CatBoost
