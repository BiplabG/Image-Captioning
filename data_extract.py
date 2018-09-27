# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:05:55 2018

@author: bipla
"""

import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#Load the annotation file
fp = open('annotations/captions_val2017.json', "r")
"""Load the json file in to a python variable. 
Creates a dictionary of four keys info, images, annotations, licenses"""
allData = json.load(fp) 

imageData = allData['images'] #Contains list of image information
annData = allData['annotations'] #Contains list of caption information

def extract(down, up):
    """Returns this 2D array with image information in the format:
        image_id, file_name, height, width, captions (5 captions)."""
    image_caption_array = [] 
    for i in range(down, up):
        information = [] #List that contains the image information
        image_instance = imageData[i]
        image_id = image_instance['id']
        information.append(image_id)#Zero the image_id
        information.append(image_instance['file_name']) #First the image file_name
        information.append(image_instance['height']) #Second the height of the image
        information.append(image_instance['width']) #Third the width of the image
        for caption in annData:
            if caption['image_id'] == image_id:
                information.append(caption['caption']) #This appends the five captions
        #Insert the information list to the dictionary
        image_caption_array.append(information)
        
        if (i%20 == 0):
            print(i, " items completed extracting.")
        
    return image_caption_array

#data_extracted = extract(600, 750) #Extract the 1000 first data
#data_extracted = np.array(data_extracted) #Convert to numpy array
#np.save('data_extracted/data_extracted_for_loss.npy', data_extracted)

#b = np.load('data_extracted_validation.npy')
#print(b)
#
#a = extract(245,249)
#image_dir = '/media/biplab/Drive/coco/train2017/'
#for i in a:
#    path = image_dir + i[1]
#    img = Image.open(path)
#    plt.imshow(img)
#    print(i[0])
#    print(i[4], i[5], i[6], i[7], i[8])
    
    