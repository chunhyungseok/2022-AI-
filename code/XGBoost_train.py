#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.metrics import f1_score
import optuna
from xgboost import XGBClassifier
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
import Data_processing as dp
import joblib


# In[2]:


train_path = '../data/train.csv'
model_path = '../model/XGBoost.pkl'


# In[3]:


#train 데이터 불러오기
train = pd.read_csv(train_path)


# In[4]:


dic_disease_type = dp.get_dic(train['disease_type'])
dic_disease_state = dp.get_dic(train['disease_state'])
train['disease_type'] = train['disease_type'].map(lambda a: dp.dic_except(dic_disease_type, a))
train['disease_state'] = train['disease_state'].map(lambda a: dp.dic_except(dic_disease_state, a))


# In[5]:


train, val = train_test_split(train, train_size=0.8, random_state=12)

train_protein_features, train_epitope_features, train_disease_features, train_label_list = dp.get_preprocessing('train', train)
val_protein_features, val_epitope_features, val_disease_features, val_label_list = dp.get_preprocessing('val', val)


# In[6]:


train_protein_features = np.array(train_protein_features)
train_epitope_features = np.array(train_epitope_features)
train_disease_features = np.array(train_disease_features)
X_train = np.concatenate((train_protein_features, train_epitope_features, train_disease_features), axis=1)
y_train = np.array(train_label_list)

val_protein_features = np.array(val_protein_features)
val_epitope_features = np.array(val_epitope_features)
val_disease_features = np.array(val_disease_features)
X_val = np.concatenate((val_protein_features, val_epitope_features, val_disease_features), axis=1)
y_val = np.array(val_label_list)


# In[7]:


train = pd.read_csv(train_path)
dic_disease_type = dp.get_dic(train['disease_type'])
dic_disease_state = dp.get_dic(train['disease_state'])
train['disease_type'] = train['disease_type'].map(lambda a: dp.dic_except(dic_disease_type, a))
train['disease_state'] = train['disease_state'].map(lambda a: dp.dic_except(dic_disease_state, a))
protein_features, epitope_features, disease_features, label_list = dp.get_preprocessing('train', train)
protein_features = np.array(protein_features)
epitope_features = np.array(epitope_features)
disease_features = np.array(disease_features)
X = np.concatenate((protein_features, epitope_features, disease_features), axis=1)
y = np.array(label_list)


# In[ ]:


def objective(trial: Trial) -> float:
    param = {'verbosity':1,
             'objective':'binary:logistic', #
             'max_depth':trial.suggest_int('max_depth',3,30),
             'learning_rate':trial.suggest_loguniform('learning_rate',1e-8,1e-2),
             'n_estimators':trial.suggest_int('n_estimators',100,3000),
             'subsample':trial.suggest_loguniform('subsample',0.7,1),
             'min_child_weight': trial.suggest_int('min_child_weight', 1, 300 ),
             'alpha': trial.suggest_loguniform( 'alpha', 1e-3, 10.0),
             'random_state': 42}
    model = XGBClassifier(**param)
    model.fit(
        np.array(X_train),
        np.array(y_train),
        eval_set=[(np.array(X_train), np.array(y_train)), (np.array(X_val), np.array(y_val))],
        early_stopping_rounds=100,
        eval_metric = 'logloss',
        verbose=False
    )

    pred = model.predict(np.array(X_val))
    log_score = log_loss(np.array(y_val), pred)
    
    return log_score

sampler = TPESampler(seed=42)
studyXGB = optuna.create_study(

    study_name="XGboost",
    direction="minimize",
    sampler=sampler,
)

studyXGB.optimize(objective, n_trials=100)
print("Best Score:", studyXGB.best_value)
print("Best trial:", studyXGB.best_trial.params)


# #### Best trial: {'max_depth': 16, 'learning_rate': 0.0025640961747953163, 'n_estimators': 1546, 'subsample': 0.7352711832885261, 'min_child_weight': 1, 'alpha': 0.027937140843060745}

# In[8]:


# 탐색한 최적의 하이퍼 파라미터 기반 전체 train dataset을 이용하여 XGBoost train
model = XGBClassifier(**studyXGB.best_trial.params)
model.fit(
    np.array(X),
    np.array(y),
    eval_set=[(np.array(X), np.array(y))],
    early_stopping_rounds=100,
    eval_metric = 'logloss',
    verbose=False,
)


# In[12]:


joblib.dump(model, model_path)


# In[ ]:




