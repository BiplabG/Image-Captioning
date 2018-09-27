#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:40:07 2018

@author: biplab
"""

import torch
import torch.nn as nn
import torchvision.models as models

def to_var(x, requires_grad = True):
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.tensor(x, requires_grad = requires_grad)

class EncoderCNN(nn.Module):
    def __init__(self):
        """This loads the pretrained resnet-152 and replaces the top fully connected layer"""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained = True)
        modules = list(resnet.children())[:-3]
        #This removes the last layer and second last layer
        self.resnet = nn.Sequential(*modules)
        
        
    def forward(self, images):
        """Extract feature vectors from input images"""
        with torch.no_grad():
            features = self.resnet(images)
#            print("Resnet output size:",features.size())
            #Convert feature to size compatible to decoder
            features = features.view(1, 1024, -1)
            features = features.transpose(1, 2)
#            print("Final CNN output size:", features.size())
            #This returns features of size 196*1024
        return features
        
class DecoderRNN(nn.Module):
    def __init__(self, in_dim, vis_dim, vis_num, embed_size, hidden_size, vocab_size, num_layers, max_seq_length = 20):
        """Set the hyper-parameters and layers for decoder"""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + vis_dim, hidden_size, num_layers) #Write detailly
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        self.max_seg_length = max_seq_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.in_dim = in_dim
        self.vis_dim = vis_dim
        self.vis_num = vis_num
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #Initial Linear layer to map features from B*1024*196 to B*512*196
        self.start_linear = nn.Linear(self.in_dim, self.vis_dim, bias=False)
        
        #attention part #Write detailly
        self.att_vw = nn.Linear(self.vis_dim, self.vis_dim, bias = False) #Linear model for input features
        self.att_hw = nn.Linear(self.hidden_size, self.vis_dim, bias = False) #Linear model for hidden inputs
        self.att_bias = nn.Parameter((torch.zeros(vis_num))) #bias input
        self.att_w = nn.Linear(self.vis_dim, 1, bias=False) #Linear model before softmax
        
    def attention_layer(self, features, hiddens):
        """
        :param features : batch_size * 196 * 512
        :param hiddens  : batch_size * hidden_dim
        :returns the context, alpha
        """
        att_fea = self.att_vw(features)
#        print(att_fea.size())
        att_h = self.att_hw(hiddens).unsqueeze(1)
#        print(att_h.size())
#        print(self.att_bias.view(1, -1, 1).size())
        att_full = nn.ReLU()(att_fea + att_h + self.att_bias.view(1, -1, 1))
#        print(att_full.size())
        att_out = self.att_w(att_full).squeeze(2)
#        print(att_out.size())
        
        alpha = nn.Softmax()(att_out)
#        print(alpha.unsqueeze(2).size())
#        print((features*alpha.unsqueeze(2)).size())
        context = torch.sum(features*alpha.unsqueeze(2), 1)
#        print(context.size())
        return context, alpha
    
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions"""
        
        batch_size, time_step = captions.data.shape
        
        embeddings = self.embed(captions)
        features = self.start_linear(features)
        feas = torch.mean(features, 1)#batch_size * 512
        h0, c0 = self.get_start_states(batch_size)
        
        predicts = torch.zeros(batch_size, time_step, self.vocab_size)
        
#        print(lengths)
        for step in range(time_step):
            batch_size = sum(i >= step for i in lengths)
            if step != 0:
                feas, alpha = self.attention_layer(features[:batch_size, :], h0[:batch_size, :])
            words = (embeddings[:batch_size, step, :]).squeeze(1)
#           print(feas.shape, words.shape)
            inputs = torch.cat([feas, words], 1) #Look why is it done?
            h0, c0 = self.lstm(inputs, (h0[:batch_size, :], c0[:batch_size, :]))
            outputs = self.fc_out(h0)
            predicts[:batch_size, step, :] = outputs
#            print(step)
#            print(outputs)
#            print(outputs.shape)
#            print(predicts.shape)
        return predicts
    
    def get_start_states(self, batch_size):
        h0 = to_var(torch.zeros(batch_size, self.hidden_size))
        c0 = to_var(torch.zeros(batch_size, self.hidden_size))
        return h0, c0
    
    def sample(self, features, states=None):
        """This generates captions for given image features using greedy search"""
        sampled_ids = []
        alphas = []
        batch_size = features.shape[0]
        
        words = self.embed(to_var(torch.ones(batch_size, 1).long())).squeeze(1)
        print(words.size())
        h0, c0 = self.get_start_states(batch_size)
        
        features = self.start_linear(features)
        feas = torch.mean(features, 1) #size batch_size*512
        
        for step in range(self.max_seg_length):
            if step != 0:
                feas, alpha = self.attention_layer(features, h0)
                alphas.append(alpha)
            
            inputs = torch.cat([feas, words], 1)
            h0, c0 = self.lstm(inputs, (h0, c0))
            
            outputs = self.fc_out(h0)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.unsqueeze(1))
            words = self.embed(predicted)
            
        sampled_ids = torch.cat(sampled_ids, 1)
        return sampled_ids.squeeze(), alphas
