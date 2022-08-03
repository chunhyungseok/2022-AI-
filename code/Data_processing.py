#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# In[2]:


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


# In[3]:


#Unknown 토큰 처리
def dic_except(dic, a):
    try:
        return dic[a]
    except:
        return dic['Unknown']
    
# Antigen Features
def get_protein_feature(seq):
    protein_feature = []
    protein_feature.append(ProteinAnalysis(seq).isoelectric_point())
    protein_feature.append(ProteinAnalysis(seq).aromaticity())
    protein_feature.append(ProteinAnalysis(seq).gravy())
    protein_feature.append(ProteinAnalysis(seq).instability_index())
    return protein_feature


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


# In[5]:


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

