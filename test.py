#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 18:11:02 2018

@author: biplab
"""
import numpy as np
import os
import torch
import pickle
from build_vocab import Vocabulary

#vocab = pickle.load(open('vocab.pkl', 'rb'))
#
#data = torch.load('features/batch_1.pt')
#features = []
#for item in data:
#    caption = []
#    for token in item[2]:
#        word = vocab.idx2word[token.item()]
#        caption.append(word)
#    features.append(caption)
#    
#print(features)

from torch.nn.utils.rnn import pad_sequence

data = torch.load('features/batch_1.pt')
#Making features, targets and lengths from features saved
features = []
targets = []
for item in data:
    features.append(item[1])
    targets.append(item[2])
lengths = []
for i in targets:
    lengths.append(len(i))

print(lengths)
t = pad_sequence(targets, batch_first = True)
f = pad_sequence(features, batch_first= True)
print(f)
    
#print(features)
#print(targets)
#print(torch.tensor(features))


#folder = 'features'
#
#for filename in os.listdir(folder):
#    print(filename)

