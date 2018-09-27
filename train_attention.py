#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 23:00:46 2018

@author: biplab
"""

import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import pickle
import os

from attention_model import DecoderRNN
from build_vocab import Vocabulary

#This is the train function that trains the decoder
def train(feature_dir, model, optimizer, criterion, epoch, total_epoch):
    log_array = []
    num_batches = len(os.listdir(feature_dir))    
    for i, filename in enumerate(os.listdir(feature_dir)):
        data = torch.load(feature_dir+filename)
        #Making features, targets and lengths from features saved
        features = []
        targets = []
        for item in data:
            features.append(item[1])
            targets.append(item[2])
        lengths = []
        for j in targets:
            lengths.append(len(j))
            
        targets = torch.tensor(pad_sequence(targets, batch_first=True))
        features = torch.tensor(pad_sequence(features, batch_first=True), requires_grad = True)
        features = features.squeeze(1)
#        print(features.shape)

        optimizer.zero_grad()
        model.zero_grad()
        predicts = model(features, targets[:, :-1], [l-1 for l in lengths])
#        print(predicts[2])
        predicts = pack_padded_sequence(predicts[:, :], [l-1 for l in lengths], batch_first = True)[0]
        targets = pack_padded_sequence(targets[:, 1:], [l-1 for l in lengths], batch_first = True)[0]
#        print(predicts.size())
#        print(targets.size())
        loss = criterion(predicts, targets)
#        print(loss)
        loss.backward()
        optimizer.step()
        
        log = open('logs/training_log_phase.txt', 'a')
        print("Batch: ", i,"/", num_batches," is being trained.")
        log.write("Batch: "+str(i)+ "/"+str(num_batches))
        print("Epoch: ", epoch,"/", total_epoch)
        log.write(" Epoch: "+str(epoch)+"/"+ str(total_epoch))
        print("Loss: ", loss, "Perplexity: ", np.exp(loss.item()))
        log.write(" Loss: "+ str(loss)+ " Perplexity: "+str(np.exp(loss.item()))+ "\n")
        print("---------------------")
        log_array.append([epoch, i, loss, np.exp(loss.item()) ])
    
    np.save('logs/log_epoch_'+str(epoch)+'.npy', log_array )
        
vocab_path = "vocab.pkl"
feature_dir = "features/"
#load vocabulary wrapper
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

#Model Setting
in_dim = 1024
vis_dim = 512
vis_num = 196
embed_dim = 512
hidden_dim = 512
vocab_size = len(vocab)
num_layers = 2
dropout_ratio = 0.5

model = DecoderRNN(in_dim = in_dim,
                   vis_dim = vis_dim,
                   vis_num = vis_num,
                   embed_size = embed_dim,
                   hidden_size = hidden_dim,
                   vocab_size = vocab_size,
                   num_layers = num_layers)

#optimizer settings
lr = 0.001 #learning rate
num_epochs = 40#Number of epochs of training for each batch
optimizer = optim.Adam(model.parameters(), lr = lr)
for name in model.named_parameters():
#    print(name)

#decoder_path = 'checkpoints/model_5.ckpt'
#model.load_state_dict(torch.load(decoder_path))
#print(torch.load(decoder_path))


#Criterion 
criterion = nn.CrossEntropyLoss()
model.cpu()
model.train()

print("Number of epochs:", num_epochs)
for epoch in range(1, num_epochs):
    train(feature_dir = feature_dir, model = model, optimizer = optimizer, criterion = criterion, epoch = epoch, total_epoch = num_epochs)
    checkpoint_path = './checkpoints/model_last_train_' + str(epoch) +'.ckpt'
    torch.save(model.state_dict(), checkpoint_path)
    