#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 17:00:09 2018

@author: biplab
"""

import nltk
import pickle
from collections import Counter
import numpy as np

class Vocabulary(object):
    """Simple vocabulary wrapper"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)
    
def build_vocab(data_all, threshold):
    
    counter = Counter()
    for i, item in enumerate(data_all):
        caption = item[4] + " " + item[5] + " " + item[6] + " " + item[7] + " " + item[8] + " "
        tokens = nltk.tokenize.word_tokenize((caption.lower()))
        counter.update(tokens)
        
        if (i+1) % 10 == 0:
                print("[{}] Tokenized the captions.".format(i+1))
    
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    
    #Create a vocab wrapper and add some special tokens
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    
    for word in words:
        vocab.add_word(word)
        
    return vocab

#data_extracted = np.load('data_extracted/data_extracted.npy')
#vocab = build_vocab(data_extracted, 3)
#
#vocab_path = 'vocab.pkl'
#with open(vocab_path, 'wb') as f:
#    pickle.dump(vocab, f)
#
##vocab = pickle.load(open(vocab_path, 'rb'))
#
#print("Total vocabulary size:", len(vocab))
