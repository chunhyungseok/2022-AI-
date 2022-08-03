#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from tqdm import tqdm
from sklearn.metrics import f1_score

from Bio.SeqUtils.ProtParam import ProteinAnalysis

import optuna
from xgboost import XGBClassifier
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
from tqdm import tqdm


# # 전체 코드 process
# 
#     1. 데이터 전 처리
#         - disease_dic 생성, train 과정에 없는 disease들은 0으로 unknown 토큰 처리
#         - 결측치 train 데이터 기반으로 채우기 및 정규화
#         
#     2. Input_feautres
#         - Antigen eatures
#         - Peptide Epitope features
#         - Disease features
#         
#     3. XGBoost 
#         - Optuna 기반 최적의 하이퍼 파라미터 탐색
#         - Feature importance 계산

# In[ ]:


train_path = '../data/train.csv'


# # Data preprocessing

# In[5]:


#train 데이터 불러오기
train = pd.read_csv(train_path)


# # Disease Features

# In[6]:


# disease dic 생성
def get_dic(data):
    vocab = {}
    for name in data:
        if name not in vocab:
            vocab[name]=0
        vocab[name] += 1
    vocab_sorted = sorted(vocab.items(), key=lambda x:x[1], reverse=True)
    token_dic = {}
    i = 1
    # train에서 보지 않은 disease들 Unknown 토큰 처리
    # train과정에서 없는 disease들은 0으로 처리하였음
    token_dic['Unknown'] = 0
    for (name, freq) in vocab_sorted:
        token_dic[name] = i
        i += 1
    return token_dic

def dic_except(dic, a):
    try:
        return dic[a]
    except:
        return dic['Unknown']

dic_disease_type = get_dic(train['disease_type'])
dic_disease_state = get_dic(train['disease_state'])

train['disease_type'] = train['disease_type'].map(lambda a: dic_except(dic_disease_type, a))
train['disease_state'] = train['disease_state'].map(lambda a: dic_except(dic_disease_state, a))


# # Epitope Peptide Features

# In[7]:


# 총 7가지 peptide features를 계산

def get_peptide_feature(seq): # CTD descriptor
    CTD = {'hydrophobicity': {1: ['R', 'K', 'E', 'D', 'Q', 'N'], 2: ['G', 'A', 'S', 'T', 'P', 'H', 'Y'], 3: ['C', 'L', 'V', 'I', 'M', 'F', 'W']},
           'normalized.van.der.waals': {1: ['G', 'A', 'S', 'T', 'P', 'D', 'C'], 2: ['N', 'V', 'E', 'Q', 'I', 'L'], 3: ['M', 'H', 'K', 'F', 'R', 'Y', 'W']},
           'polarity': {1: ['L', 'I', 'F', 'W', 'C', 'M', 'V', 'Y'], 2: ['P', 'A', 'T', 'G', 'S'], 3: ['H', 'Q', 'R', 'K', 'N', 'E', 'D']},
           'polarizability': {1: ['G', 'A', 'S', 'D', 'T'], 2: ['C', 'P', 'N', 'V', 'E', 'Q', 'I', 'L'], 3: ['K', 'M', 'H', 'F', 'R', 'Y', 'W']},
           'charge': {1: ['K', 'R'], 2: ['A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'], 3: ['D', 'E']},
           'secondary': {1: ['E', 'A', 'L', 'M', 'Q', 'K', 'R', 'H'], 2: ['V', 'I', 'Y', 'C', 'W', 'F', 'T'], 3: ['G', 'N', 'P', 'S', 'D']},
           'solvent': {1: ['A', 'L', 'F', 'C', 'G', 'I', 'V', 'W'], 2: ['R', 'K', 'Q', 'E', 'N', 'D'], 3: ['M', 'S', 'P', 'T', 'H', 'Y']}}
    
    seq = str(seq)
    sequencelength = len(seq)
    Sequence_group = []
    
    for AAproperty in CTD:
        propvalues = ""
        for letter in seq:
            if letter in CTD[AAproperty][1]:
                propvalues += "1"
            elif letter in CTD[AAproperty][2]:
                propvalues += "2"
            elif letter in CTD[AAproperty][3]:
                propvalues += "3"
        abpos_1 = [i for i in range(len(propvalues)) if propvalues.startswith("1", i)]
        abpos_1 = [x+1 for x in abpos_1]
        abpos_1.insert(0, "-")
        abpos_2 = [i for i in range(len(propvalues)) if propvalues.startswith("2", i)]
        abpos_2 = [x+1 for x in abpos_2]
        abpos_2.insert(0, "-")
        abpos_3 = [i for i in range(len(propvalues)) if propvalues.startswith("3", i)]
        abpos_3 = [x+1 for x in abpos_3]
        abpos_3.insert(0, "-")
        property_group1_length = propvalues.count("1")
        
        if property_group1_length == 0:
            Sequence_group.extend([0, 0, 0, 0, 0])
        elif property_group1_length == 1:
            Sequence_group.append((abpos_1[1]/sequencelength)*100)
            Sequence_group.append((abpos_1[1]/sequencelength)*100)
            Sequence_group.append((abpos_1[1]/sequencelength)*100)
            Sequence_group.append((abpos_1[1]/sequencelength)*100)
            Sequence_group.append((abpos_1[1]/sequencelength)*100)
        elif property_group1_length == 2:
            Sequence_group.append((abpos_1[1]/sequencelength)*100)
            Sequence_group.append((abpos_1[1]/sequencelength)*100)
            Sequence_group.append((abpos_1[round((0.5*property_group1_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_1[round((0.75*property_group1_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_1[property_group1_length]/sequencelength)*100)
        else:
            Sequence_group.append((abpos_1[1]/sequencelength)*100)
            Sequence_group.append((abpos_1[round((0.25*property_group1_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_1[round((0.5*property_group1_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_1[round((0.75*property_group1_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_1[property_group1_length]/sequencelength)*100)

        property_group2_length = propvalues.count("2")
        if property_group2_length == 0:
            Sequence_group.extend([0, 0, 0, 0, 0])
        elif property_group2_length == 1:
            Sequence_group.append((abpos_2[1]/sequencelength)*100)
            Sequence_group.append((abpos_2[1]/sequencelength)*100)
            Sequence_group.append((abpos_2[1]/sequencelength)*100)
            Sequence_group.append((abpos_2[1]/sequencelength)*100)
            Sequence_group.append((abpos_2[1]/sequencelength)*100)
        elif property_group2_length == 2:
            Sequence_group.append((abpos_2[1]/sequencelength)*100)
            Sequence_group.append((abpos_2[1]/sequencelength)*100)
            Sequence_group.append((abpos_2[round((0.5*property_group2_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_2[round((0.75*property_group2_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_2[property_group2_length]/sequencelength)*100)
        else:
            Sequence_group.append((abpos_2[1]/sequencelength)*100)
            Sequence_group.append((abpos_2[round((0.25*property_group2_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_2[round((0.5*property_group2_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_2[round((0.75*property_group2_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_2[property_group2_length]/sequencelength)*100)

        property_group3_length = propvalues.count("3")
        if property_group3_length == 0:
            Sequence_group.extend([0, 0, 0, 0, 0])
        elif property_group3_length == 1:
            Sequence_group.append((abpos_3[1]/sequencelength)*100)
            Sequence_group.append((abpos_3[1]/sequencelength)*100)
            Sequence_group.append((abpos_3[1]/sequencelength)*100)
            Sequence_group.append((abpos_3[1]/sequencelength)*100)
            Sequence_group.append((abpos_3[1]/sequencelength)*100)
        elif property_group3_length == 2:
            Sequence_group.append((abpos_3[1]/sequencelength)*100)
            Sequence_group.append((abpos_3[1]/sequencelength)*100)
            Sequence_group.append((abpos_3[round((0.5*property_group3_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_3[round((0.75*property_group3_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_3[property_group3_length]/sequencelength)*100)
        else:
            Sequence_group.append((abpos_3[1]/sequencelength)*100)
            Sequence_group.append((abpos_3[round((0.25*property_group3_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_3[round((0.5*property_group3_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_3[round((0.75*property_group3_length)-0.1)]/sequencelength)*100)
            Sequence_group.append((abpos_3[property_group3_length]/sequencelength)*100)
    return Sequence_group


# # Antigen Features

# In[ ]:


# Bio python 툴을 이용하여 Antigen의 4가지 features extraction

def get_protein_feature(seq):
    protein_feature = []
    protein_feature.append(ProteinAnalysis(seq).isoelectric_point())
    protein_feature.append(ProteinAnalysis(seq).aromaticity())
    protein_feature.append(ProteinAnalysis(seq).gravy())
    protein_feature.append(ProteinAnalysis(seq).instability_index())
    return protein_feature


# # Get Features

# In[8]:



def get_preprocessing(data_type, new_df):   
    protein_features = []
    epitope_features = []
    disease_features = []
        
    for epitope, antigen, d_type, d_state in tqdm(zip(new_df['epitope_seq'], new_df['antigen_seq'], new_df['disease_type'], new_df['disease_state'])):        

        protein_features.append(get_protein_feature(antigen))
        epitope_features.append(get_peptide_feature(epitope))
        disease_features.append([d_type, d_state])
    
    label_list = None
    if data_type != 'test':
        label_list = []
        for label in new_df['label']:
            label_list.append(label)
    print(f'{data_type} dataframe preprocessing was done.')
    return protein_features, epitope_features, disease_features, label_list


# # Train & Validation Split

# In[9]:


train, val = train_test_split(train, train_size=0.8, random_state=12)

train_protein_features, train_epitope_features, train_disease_features, train_label_list = get_preprocessing('train', train)
val_protein_features, val_epitope_features, val_disease_features, val_label_list = get_preprocessing('val', val)


# In[10]:


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


# # XGBoost Optimization

# In[14]:


# XGBoost 머신 러닝 모델 하이퍼 파라미터 최적화

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


# ### Best trial: {'max_depth': 16, 'learning_rate': 0.0025640961747953163, 'n_estimators': 1546, 'subsample': 0.7352711832885261, 'min_child_weight': 1, 'alpha': 0.027937140843060745}
