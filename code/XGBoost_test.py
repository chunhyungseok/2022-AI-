#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from tqdm import tqdm
from xgboost import XGBClassifier
import Data_processing as dp
import joblib


# In[6]:


train_path = '../data/train.csv'
test_path = '../data/test.csv'
submit_sample_path = '../data/sample_submission.csv'
submit_path = '../data/XGBoost.csv'
model_path = '../model/XGBoost_opt.pkl'


# In[3]:


train = pd.read_csv(train_path)
dic_disease_type = dp.get_dic(train['disease_type'])
dic_disease_state = dp.get_dic(train['disease_state'])


# In[4]:


# test dataset 불러오기

test_df = pd.read_csv(test_path)
#test data의 disease들 중 train과정에서 보지 않은 disease는 'unknown' 토큰인 0으로 처리
test_df['disease_type'] = test_df['disease_type'].map(lambda a: dp.dic_except(dic_disease_type, a))
test_df['disease_state'] = test_df['disease_state'].map(lambda a: dp.dic_except(dic_disease_state, a))

test_protein_features, test_epitope_features, test_disease_features, label_list = dp.get_preprocessing('test', test_df)

test_protein_features = np.array(test_protein_features)
test_epitope_features = np.array(test_epitope_features)
test_disease_features = np.array(test_disease_features)
X_test = np.concatenate((test_protein_features, test_epitope_features, test_disease_features), axis=1)


# In[7]:


#load model
model = joblib.load(model_path)


# In[8]:


# test dataset을 이용하여 최종 Inference

preds_all = model.predict(np.array(X_test))
submit = pd.read_csv(submit_sample_path)
submit['label'] = preds_all
submit.to_csv(submit_path, index=False)
print('Done.')


# In[ ]:




