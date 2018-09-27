#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 15:39:51 2018

@author: biplab
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from torchvision import transforms

from build_vocab import Vocabulary
from attention_model import EncoderCNN, DecoderRNN
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform = None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

"""Argument Variables"""
vocab_path = 'vocab.pkl'
#image_path = 'example.jpg'
decoder_path = 'checkpoints/'

validation_data = np.load('data_extracted/data_extracted_validation.npy')
image_dir = 'val2017/'   

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))])

#Load vocabulary
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

encoder = EncoderCNN()
decoder = DecoderRNN(in_dim = in_dim,
                   vis_dim = vis_dim,
                   vis_num = vis_num,
                   embed_size = embed_dim,
                   hidden_size = hidden_dim,
                   vocab_size = vocab_size,
                   num_layers = num_layers)



#Prepare the image
#print(len(validation_data))
def validate_data(checkpoint_name, result_name):
    #load the trained decoder model
    #print(torch.load(decoder_path))
    decoder.load_state_dict(torch.load(decoder_path + checkpoint_name, map_location = 'cpu')["state_dict"])
    result_list = []
    for image_instance in validation_data:
        try:
            image = image_instance[1]
            image_path = image_dir + image
            
            image = load_image(image_path, transform)
            image_tensor = image.to(device)
            
            #Generate caption from the image
            feature = encoder(image_tensor)
            sampled_ids = decoder.sample(feature)
            sampled_ids = sampled_ids[0].cpu().numpy()
            
            #Convert word_ids into words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
                
            sentence = ' '.join(sampled_caption)
            #Print out the image and the generated caption
            sentence = sentence.replace('<end>', '')
            print(sentence)
            output_dict = {'image_id':image_instance[0], 'caption':sentence}
            result_list.append(output_dict)
            
            print(image_path)
        #    print("Actual Captions: ", image_instance[4], image_instance[5], image_instance[6], image_instance[7], image_instance[8])
        #        image = Image.open(image_path)
            print(len(result_list))
        except RuntimeError:
            print("got runtime error")
            pass
    
    import json
    with open('results/' + result_name, 'w') as outfile:
        json.dump(result_list, outfile)

checkpoint_list = ['model_cuda_0.ckpt',
               'model_cuda_1.ckpt',
               'model_cuda_2.ckpt',
               'model_cuda_3.ckpt',
               'model_cuda_4.ckpt',
               ]

result_list = ['result_epoch_0.json',
               'result_epoch_1.json',
               'result_epoch_2.json',
               'result_epoch_3.json',
               'result_epoch_3.json',
               ]

for i in range(5):
    validate_data(checkpoint_list[i], result_list[i])
