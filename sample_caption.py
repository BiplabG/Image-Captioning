#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 18:19:00 2018

@author: biplab
"""

import torch
#import matplotlib.pyplot as plt
#import numpy as np
import pickle
#import os
from torchvision import transforms
from PIL import Image
#import skimage.transform

from build_vocab import Vocabulary
from attention_model import EncoderCNN, DecoderRNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform = None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

#def attention_visualization(root, image_name, caption, alphas):
#    image = Image.open(os.path.join(root, image_name))
#    image = image.resize([224, 224], Image.LANCZOS)
#    plt.subplot(4,5,1)
#    plt.imshow(image)
#    plt.axis('off')
#    
#    words = caption[1:]
#    for t in range(0, len(words)):
#        if t > 18:
#            break
#        plt.subplot(4, 5, t+2)
#        plt.text(0, 1, '%s'%(words[t]) , color='black', backgroundcolor='white', fontsize=8)
#        plt.imshow(image)
##        print(alphas)
#        alp_curr = alphas[t][:].view(14, 14)
#        alp_curr = np.transpose(alp_curr.detach())
#        alp_img = skimage.transform.pyramid_expand(alp_curr.detach().numpy(), upscale=16, sigma=20)
#        plt.imshow(alp_img, alpha=0.85)
#        plt.axis('off')
#    plt.show()

"""Argument Variables"""

def get_caption(image_path):
    vocab_path = '/home/biplab/major_project/vocab.pkl'
    #image_path = 'example.jpg'
    decoder_path = '/home/biplab/major_project/checkpoints/model_cuda_60.ckpt'
    
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
    
    encoder = EncoderCNN()
    decoder = DecoderRNN(in_dim = in_dim,
                       vis_dim = vis_dim,
                       vis_num = vis_num,
                       embed_size = embed_dim,
                       hidden_size = hidden_dim,
                       vocab_size = vocab_size,
                       num_layers = num_layers)
    
    #load the trained decoder model
    #print(torch.load(decoder_path))
    decoder.load_state_dict(torch.load(decoder_path, map_location = 'cpu')['state_dict'])
    
    image_val = load_image(image_path, transform)
    image_tensor = image_val.to(device)
    
    #Generate caption from the image
    feature = encoder(image_tensor)
    (sampled_ids, alphas) = decoder.sample(feature)
    sampled_ids = sampled_ids.cpu().numpy()
    
    #Convert word_ids into words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
        
    sentence = ' '.join(sampled_caption)
    return sentence

#print(get_caption('ex.jpg'))