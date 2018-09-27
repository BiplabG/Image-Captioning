#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:06:38 2018

@author: biplab
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import pickle
import nltk
import time

from build_vocab import Vocabulary
from attention_model import EncoderCNN

vocab_path = 'vocab.pkl'


def transform_image():
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))]) #Find out why
    return transform

def load_image(image_path, transform = None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS) #Find out what is image LancZos
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def sort_array(input_arr):
    return sorted(input_arr, key = lambda x : len(x[2]), reverse=True)

def caption_to_token(caption):
    vocab = pickle.load(open(vocab_path, 'rb'))
    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
    target = []
    target.append(vocab('<start>'))
    target.extend([vocab(word) for word in tokens])
    target.append(vocab('<end>'))
    target = torch.tensor(target)
    return target

def extract_features(root, transform, batch_size):
    final_array = []
    batch_count = 1
    init_time = time.time()
    for i, item in enumerate(data_extracted):
        try:
            image_path = root + item[1]
            img_tensor = load_image(image_path, transform)
            encoder = EncoderCNN()
            feas = encoder(img_tensor).cpu()
        except OSError:
            f = open('corrupt.txt', 'a')
            print("File not found error")
            f.write(image_path)
            f.close()
            continue
        except RuntimeError:
            g = open('logs_fea_ext_error.txt', 'a')
            g.write(image_path)
            g.close()
            print("Runtime Error occurred.")
            continue
        
        if i%10 == 0:
            print(i, " images completed")
        for j in range(4, 9):
            tokenized_caption = caption_to_token(item[j])
            arr = [image_path, feas.data, tokenized_caption]
            final_array.append(arr)
                        
            if len(final_array) == batch_size:
                #For time calculation purpose
                new_time = time.time()
                print("Time taken:", new_time-init_time)
                init_time = new_time
                
                feature_file_name = "features/batch_35_40k_"+str(batch_count)+".pt"
                print(batch_count," batches completed.")
                final_array = sort_array(final_array)
                torch.save(final_array, feature_file_name)
                batch_count += 1
                final_array = []
            

        
def extract_features_for_loss(root, transform, batch_size, data_extracted):
    final_array = []
    batch_count = 1
    init_time = time.time()
    for i, item in enumerate(data_extracted):
        try:
            image_path = root + item[1]
            img_tensor = load_image(image_path, transform)
            encoder = EncoderCNN()
            feas = encoder(img_tensor).cpu()
        except OSError:
            f = open('corrupt.txt', 'a')
            print("File not found error")
            f.write(image_path)
            f.close()
            continue
        except RuntimeError:
            g = open('logs_fea_ext_error.txt', 'a')
            g.write(image_path)
            g.close()
            print("Runtime Error occurred.")
            continue
        
        if i%10 == 0:
            print(i, " images completed")
        for j in range(4, 9):
            tokenized_caption = caption_to_token(item[j])
            arr = [image_path, feas.data, tokenized_caption]
            final_array.append(arr)
                        
            if len(final_array) == batch_size:
                #For time calculation purpose
                new_time = time.time()
                print("Time taken:", new_time-init_time)
                init_time = new_time
                
                feature_file_name = "features/batch_100_for_loss_"+str(batch_count)+".pt"
                print(batch_count," batches completed.")
                final_array = sort_array(final_array)
                torch.save(final_array, feature_file_name)
                batch_count += 1
                final_array = []
                
#data_extracted = np.load('data_extracted/data_extracted_for_loss.npy')
#image_dir = 'val2017/'        
#        
#transform = transform_image()
#
#extract_features_for_loss(image_dir, transform, 196, data_extracted)