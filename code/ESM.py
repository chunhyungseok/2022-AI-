#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings(action='ignore')
from transformers import ESMTokenizer, ESMForMaskedLM, ESMModel
from sklearn.model_selection import train_test_split


# # Parameter

# In[2]:


device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')


# In[3]:


CFG = {
    'NUM_WORKERS':4,
    'ANTIGEN_WINDOW':256,
    'ANTIGEN_MAX_LEN':512, 
    'EPITOPE_MAX_LEN':512,
    'EPOCHS':50,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':128,
    'THRESHOLD':0.5,
    'SEED':41
}


# In[5]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정


# In[6]:


tokenizer = ESMTokenizer.from_pretrained("facebook/esm-1b", do_lower_case=False)


# # Preprocessing

# In[7]:


def seqtoinput(seq):
    for j in range(len(seq)-1):
        seq = seq[:j+j+1]+ ' ' + seq[j+j+1:]
    return seq
    

def get_preprocessing(data_type, new_df, tokenizer):
    epitope_ids_list = []
    epitope_mask_list = []
    
    antigen_ids_list = []
    antigen_mask_list = []
        
    for epitope, antigen, s_p, e_p in tqdm(zip(new_df['epitope_seq'], new_df['antigen_seq'], new_df['start_position'], new_df['end_position'])):        
        # Left antigen : [start_position-WINDOW : start_position]
        # Right antigen : [end_position : end_position+WINDOW]
        mean = int((s_p+e_p)/2)
        start_position = mean-CFG['ANTIGEN_WINDOW']-1
        end_position = mean+CFG['ANTIGEN_WINDOW']
        if start_position < 0:
            start_position = 0
        if end_position > len(antigen):
            end_position = len(antigen)
        
        antigen = antigen[int(start_position):int(end_position)]
        # left / right antigen sequence 추출

        if CFG['EPITOPE_MAX_LEN']<len(epitope):
            epitope = epitope[:CFG['EPITOPE_MAX_LEN']]
        else:
            epitope = epitope[:]
        
        antigen = seqtoinput(antigen)
        epitope = seqtoinput(epitope)
        
        
        antigen_input = tokenizer(antigen, add_special_tokens=True, pad_to_max_length=True, max_length = CFG['ANTIGEN_MAX_LEN'])
        antigen_ids = antigen_input['input_ids']
        antigen_mask = antigen_input['attention_mask']
        
        
        epitope_input = tokenizer(epitope, add_special_tokens=True, pad_to_max_length=True, max_length = CFG['EPITOPE_MAX_LEN'])
        epitope_ids = epitope_input['input_ids']
        epitope_mask = epitope_input['attention_mask']
        
        
        epitope_ids_list.append(epitope_ids)
        epitope_mask_list.append(epitope_mask)
        
        antigen_ids_list.append(antigen_ids)
        antigen_mask_list.append(antigen_mask)

    
    label_list = None
    if data_type != 'test':
        label_list = []
        for label in new_df['label']:
            label_list.append(label)
    print(f'{data_type} dataframe preprocessing was done.')
    return epitope_ids_list, epitope_mask_list, antigen_ids_list, antigen_mask_list, label_list


# In[8]:


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train, val = train_test_split(train, train_size=0.8, random_state=12)

train_epitope_ids_list, train_epitope_mask_list, train_antigen_ids_list, train_antigen_mask_list, train_label_list = get_preprocessing('train', train, tokenizer)
val_epitope_ids_list, val_epitope_mask_list, val_antigen_ids_list, val_antigen_mask_list, val_label_list = get_preprocessing('val', val, tokenizer)


# # Load Data

# In[9]:


class CustomDataset(Dataset):
    def __init__(self, epitope_ids_list, epitope_mask_list, antigen_ids_list, antigen_mask_list, label_list):
        self.epitope_ids_list = epitope_ids_list
        self.epitope_mask_list = epitope_mask_list
        self.antigen_ids_list = antigen_ids_list
        self.antigen_mask_list = antigen_mask_list

        self.label_list = label_list
        
    def __getitem__(self, index):
        self.epitope_ids = self.epitope_ids_list[index]
        self.epitope_mask = self.epitope_mask_list[index]
        
        self.antigen_ids = self.antigen_ids_list[index]
        self.antigen_mask = self.antigen_mask_list[index]
        
        
        if self.label_list is not None:
            self.label = self.label_list[index]
            return torch.tensor(self.epitope_ids), torch.tensor(self.epitope_mask), torch.tensor(self.antigen_ids), torch.tensor(self.antigen_mask), self.label
        else:
            return torch.tensor(self.epitope_ids), torch.tensor(self.epitope_mask), torch.tensor(self.antigen_ids), torch.tensor(self.antigen_mask)
        
    def __len__(self):
        return len(self.epitope_ids_list)


# In[10]:


train_dataset = CustomDataset(train_epitope_ids_list, train_epitope_mask_list, train_antigen_ids_list, train_antigen_mask_list, train_label_list)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['NUM_WORKERS'])

val_dataset = CustomDataset(val_epitope_ids_list, val_epitope_mask_list, val_antigen_ids_list, val_antigen_mask_list, val_label_list)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=CFG['NUM_WORKERS'])


# # Model

# In[11]:


class TransformerModel(nn.Module):
    def __init__(self,
                 epitope_length=CFG['EPITOPE_MAX_LEN'],
                 epitope_emb_node=1024,
                 epitope_hidden_dim=1024,
                 antigen_length=CFG['ANTIGEN_MAX_LEN'],
                 antigen_emb_node=1024,
                 antigen_hidden_dim=1024,
                 pretrained_model='facebook/esm-1b'
                ):
        super(TransformerModel, self).__init__()              
        # Transformer                
        self.esm = ESMModel.from_pretrained(pretrained_model)        
        
        self.attention = nn.MultiheadAttention(embed_dim = 1280, num_heads = 8, batch_first = True)
        
        
        in_channels = 1280
            
        self.classifier = nn.Sequential(
            nn.LeakyReLU(True),
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, in_channels//4),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(in_channels//4),
            nn.Linear(in_channels//4, 1)
        )
        
    def forward(self, epitope_x1, epitope_x2, antigen_x1, antigen_x2):
        
        # Get Embedding Vector
        epitope_x = self.esm(input_ids=epitope_x1, attention_mask=epitope_x2).last_hidden_state
        
        
        antigen_x = self.esm(input_ids=antigen_x1, attention_mask=antigen_x2).last_hidden_state        
        
        
        
        # LSTM
        
        x, _ = self.attention(antigen_x, antigen_x, epitope_x)
        
        x = torch.mean(x, dim=1)
        
        
        x = self.classifier(x).view(-1)
        return x


# # Train & Validation

# In[12]:


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device) 
    
    best_val_f1 = 0
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for epitope_ids_list, epitope_mask_list, antigen_ids_list, antigen_mask_list, label in tqdm(iter(train_loader)):
            epitope_ids_list = epitope_ids_list.to(device)
            epitope_mask_list = epitope_mask_list.to(device)

            antigen_ids_list = antigen_ids_list.to(device)
            antigen_mask_list = antigen_mask_list.to(device)

            label = label.float().to(device)
            
            optimizer.zero_grad()
            
            output = model(epitope_ids_list, epitope_mask_list, antigen_ids_list, antigen_mask_list)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
            if scheduler is not None:
                scheduler.step()
                    
        val_loss, val_f1 = validation(model, val_loader, criterion, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Val F1 : [{val_f1:.5f}]')
        
        if best_val_f1 < val_f1:
            best_val_f1 = val_f1
            torch.save(model.module.state_dict(), 'model/antigen_transformer_best_model.pth', _use_new_zipfile_serialization=False)
            print('Model Saved.')
    return best_val_f1


# In[13]:


def validation(model, val_loader, criterion, device):
    model.eval()
    pred_proba_label = []
    true_label = []
    val_loss = []
    with torch.no_grad():
        for epitope_ids_list, epitope_mask_list, antigen_ids_list, antigen_mask_list, label in tqdm(iter(val_loader)):
            epitope_ids_list = epitope_ids_list.to(device)
            epitope_mask_list = epitope_mask_list.to(device)

            antigen_ids_list = antigen_ids_list.to(device)
            antigen_mask_list = antigen_mask_list.to(device)

            label = label.float().to(device)
            
            model_pred = model(epitope_ids_list, epitope_mask_list, antigen_ids_list, antigen_mask_list)
            loss = criterion(model_pred, label)
            model_pred = torch.sigmoid(model_pred).to('cpu')
            
            pred_proba_label += model_pred.tolist()
            true_label += label.to('cpu').tolist()
            
            val_loss.append(loss.item())
            
    pred_label = np.where(np.array(pred_proba_label)>CFG['THRESHOLD'], 1, 0)
    val_f1 = f1_score(true_label, pred_label, average='macro')
    return np.mean(val_loss), val_f1


# # Run

# In[14]:


model = TransformerModel()
model = nn.DataParallel(model, device_ids=[1, 0, 3, 4, 5, 6 ])
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000*CFG['EPOCHS'], eta_min=0)


# In[15]:


for name, para in model.named_parameters():
    if name in [
'module.esm.encoder.layer.31.attention.self.query.weight',
'module.esm.encoder.layer.31.attention.self.query.bias',
'module.esm.encoder.layer.31.attention.self.key.weight',
'module.esm.encoder.layer.31.attention.self.key.bias',
'module.esm.encoder.layer.31.attention.self.value.weight',
'module.esm.encoder.layer.31.attention.self.value.bias',
'module.esm.encoder.layer.31.attention.output.dense.weight',
'module.esm.encoder.layer.31.attention.output.dense.bias',
'module.esm.encoder.layer.31.attention.LayerNorm.weight',
'module.esm.encoder.layer.31.attention.LayerNorm.bias',
'module.esm.encoder.layer.31.intermediate.dense.weight',
'module.esm.encoder.layer.31.intermediate.dense.bias',
'module.esm.encoder.layer.31.output.dense.weight',
'module.esm.encoder.layer.31.output.dense.bias',
'module.esm.encoder.layer.31.LayerNorm.weight',
'module.esm.encoder.layer.31.LayerNorm.bias',
'module.esm.encoder.layer.32.attention.self.query.weight',
'module.esm.encoder.layer.32.attention.self.query.bias',
'module.esm.encoder.layer.32.attention.self.key.weight',
'module.esm.encoder.layer.32.attention.self.key.bias',
'module.esm.encoder.layer.32.attention.self.value.weight',
'module.esm.encoder.layer.32.attention.self.value.bias',
'module.esm.encoder.layer.32.attention.output.dense.weight',
'module.esm.encoder.layer.32.attention.output.dense.bias',
'module.esm.encoder.layer.32.attention.LayerNorm.weight',
'module.esm.encoder.layer.32.attention.LayerNorm.bias',
'module.esm.encoder.layer.32.intermediate.dense.weight',
'module.esm.encoder.layer.32.intermediate.dense.bias',
'module.esm.encoder.layer.32.output.dense.weight',
'module.esm.encoder.layer.32.output.dense.bias',
'module.esm.encoder.layer.32.LayerNorm.weight',
'module.esm.encoder.layer.32.LayerNorm.bias',
'module.esm.encoder.emb_layer_norm_after.weight',
'module.esm.encoder.emb_layer_norm_after.bias',
'module.esm.pooler.dense.weight',
'module.esm.pooler.dense.bias',
'module.attention.in_proj_weight',
'module.attention.in_proj_bias',
'module.attention.out_proj.weight',
'module.attention.out_proj.bias',
'module.classifier.1.weight',
'module.classifier.1.bias',
'module.classifier.2.weight',
'module.classifier.2.bias',
'module.classifier.4.weight',
'module.classifier.4.bias',
'module.classifier.5.weight',
'module.classifier.5.bias']:
        para.requires_grad = True
        
    else:
        para.requires_grad = False
        
for name, para in model.named_parameters():
    print(para)


# In[16]:


best_score = train(model, optimizer, train_loader, val_loader, scheduler, device)
print(f'Best Validation F1 Score : [{best_score:.5f}]')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




