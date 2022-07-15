#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:48:41 2022

@author: jaspers2
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



def norm_images(images,stats):
    N,C,H,W = images.shape
    
    code = stats['image_norm_code']
        
    stats_max = np.array(stats['max']).astype('float32')
    stats_min = np.array(stats['min']).astype('float32')
    stats_std = np.array(stats['std']).astype('float32')
    stats_mean = np.array(stats['mean']).astype('float32')
        
        
    max_array = stats_max.reshape(len(stats_max),1,1)
    min_array = stats_min.reshape(len(stats_min),1,1)
    std_array = stats_std.reshape(len(stats_std),1,1)
    mean_array = stats_mean.reshape(len(stats_mean),1,1)
    
    if code == []:        
        normed_images = (images-1)
    
    elif code == 1:
        normed_images = (images-min_array)/(max_array-min_array)
        
    elif code == 2:
        normed_images = (images-mean_array)/std_array
        
    elif code == 3:
        normed_images = (images-min_array)/(max_array-min_array)
        
        mean_val = []
        std_val = []
        
        for i in range(C):
            mean_val += [normed_images[:,i,:,:].mean()]
            std_val += [normed_images[:,i,:,:].std()]
        
        stats_std_2 = np.array(std_val).astype('float32')
        stats_mean_2 = np.array(mean_val).astype('float32')
            
            
        std_array_2 = stats_std_2.reshape(len(stats_std_2),1,1)
        mean_array_2 = stats_mean_2.reshape(len(stats_mean_2),1,1)
        normed_images = (normed_images-mean_array_2)/std_array_2

    
    return normed_images

def rescale_thk(thk_in,stats):
    code = stats['image_norm_code']
    thk_actual = np.array(stats['max'])
    
    if code == 1:
        raise Exception("rescale not defined")
    elif code == 2:
        raise Exception("rescale not defined")
    elif code == 3:
        raise Exception("rescale not defined")
    

def norm_labels(array,centering_factor,denom_factor):
    if type(array) != np.ndarray:
        raise Exception('norm_labels requires np array')
    

    return (array - centering_factor)/denom_factor


def rescale_labels(array, centering_factor, denom_factor):
    if type(array) != np.ndarray:
        raise Exception('rescale_labels requires np array')
        
    return array*denom_factor + centering_factor
    
    

class Network(nn.Module):
    
    def __init__(self, input_data):
        super(Network, self).__init__()
        
        self.image_stats = input_data.image_stats
        self.label_stats = input_data.label_stats
        #self.image_centering_factor = input_data.image_centering_factor
        #self.image_denom_factor = input_data.image_denom_factor
        self.label_centering_factor = input_data.label_centering_factor
        self.label_denom_factor = input_data.label_denom_factor
        self.label_names = input_data.labels_names

class Network1a(nn.Module):  
    
    def __init__(self, num_pixels, num_layers, num_labels, l1=20): #was 512               
        
        super(Network, self).__init__()
        # input image channel, output channels, square convolution kernel
        self.flatten = nn.Flatten()
        # an affine operation: y = Wx + b
        # fc implies "fully connected" because linear layers are also called fully connected
        self.fc1 = nn.Linear(31250, 100)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, num_labels)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)


    def forward(self, x):
        # (1) input layer
        x = x
        #print("shape at input: ",x.shape)
        
        # reshaping before linear layers
        #x = x.view(-1, self.num_flat_features(x))
        x = self.flatten(x)
                
        # () output layer
        # print("shape before fc1: ",x.shape)
        x = F.relu(self.fc1(x)) 
        x = self.batch_norm1(x)
        x = (self.fc2(x))
        #x = self.fc1(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# %%
class Network2(nn.Module):  
    '''conv network used for overfitting/validation of model
    
    validation data: /home/jaspers2/Documents/pixel_optimization/validation_data
    
    '''
    
    def __init__(self, num_layers, num_labels): 
        
        #kernel sizes    
        k1 = 5        
        
        #padding values
        p1 = 1
        
        out_ch_conv1 = num_layers*5

        super(Network2, self).__init__()
        # input image channel, output channels, square convolution kernel
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels = num_layers, out_channels = out_ch_conv1, kernel_size = k1, padding=p1, groups=num_layers, bias=False)  
        self.fc1 = nn.Linear(151290, 100)
        self.fc2 = nn.Linear(100, num_labels)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)

    def forward(self, x):
        x = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
#%%
class Network3(nn.Module):  
    '''
    conv network, developed after Network2, 220220
    '''
    
    def __init__(self, num_pixels_width, num_layers, num_labels):        
        #kernel sizes    
        k1 = 5
        k_pool1 = 2
        k2 = 3
        k_pool2 = 2
        k3 = 5 #was 3 only
        
        #padding values
        p1 = 1
        p2 = 1
        p3 = 0
        
        out_ch_conv1 = num_layers*5
        out_ch_conv2 = num_layers*15
        
        #out_ch_conv1 = 10
        #out_ch_conv2 = 20
        out_ch_conv3 = 25
        
        size_after_conv1 = (num_pixels_width-k1+2*p1)/1 + 1
        size_after_pool1 = np.floor(size_after_conv1/k_pool1)
        size_after_conv2 = (size_after_pool1-k2+2*p2)/1 + 1
        size_after_pool2 = int(np.floor(size_after_conv2/k_pool2))

        
                
        
        super(Network3, self).__init__()       
        
        #bias=false b/c of batchnorm use: 
        #self.conv1 = nn.Conv2d(in_channels = num_layers, out_channels = out_ch_conv1, kernel_size = k1, padding=p1)
        self.conv1 = nn.Conv2d(in_channels = num_layers, out_channels = out_ch_conv1, kernel_size = k1, padding=p1)#, groups=num_layers, bias=False)  
        self.pool1 = nn.MaxPool2d(kernel_size=k_pool1,stride=2)
        self.batch_norm1c = nn.BatchNorm2d(out_ch_conv1)
        self.conv2 = nn.Conv2d(in_channels = out_ch_conv1, out_channels = out_ch_conv2, kernel_size = k2, padding=p2)#, groups=out_ch_conv1, bias=False)
        self.pool2 = nn.MaxPool2d(k_pool2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(size_after_pool2**2*out_ch_conv2, 100)
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, num_labels)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)

    def forward(self, x):
        # (1) input layer
        x = x
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.batch_norm1c(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        #print('shape:',x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.fc2(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
#%%
class Network4(nn.Module):  
    #fully connected, experimental for pixel input
    def __init__(self, num_pixels, num_layers, num_labels, l1=20): #was 512               
        
        super(Network4, self).__init__()
        # input image channel, output channels, square convolution kernel
        self.flatten = nn.Flatten()
        # an affine operation: y = Wx + b
        # fc implies "fully connected" because linear layers are also called fully connected
        self.fc1 = nn.Linear(32, 100)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, num_labels)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)


    def forward(self, x):
        # (1) input layer
        x = x
        #print("shape at input: ",x.shape)
        
        # reshaping before linear layers
        #x = x.view(-1, self.num_flat_features(x))
        x = self.flatten(x)
                
        # () output layer
        #print("shape before fc1: ",x.shape)
        x = F.relu(self.fc1(x)) 
        x = self.batch_norm1(x)
        x = (self.fc2(x))
        #x = self.fc1(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#%%
class Network5(nn.Module):  
    '''conv network used for overfitting/validation of model
    
    validation data: /home/jaspers2/Documents/pixel_optimization/validation_data
    
    '''
    
    def __init__(self, num_pixels_width, num_layers, num_labels, kernel_size=5): 
        
        #kernel/padding values
        k1 = kernel_size       
        p1 = 1
        
        out_ch_conv1 = num_layers*20
        out_ch_conv2 = num_layers*20
        out_ch_conv3 = num_layers*20
        out_ch_conv4 = num_layers*20

        size_after_conv1 = (num_pixels_width-k1+2*p1)/1 + 1
        size_after_conv2 = (size_after_conv1-k1+2*p1)/1 + 1
        size_after_conv3 = (size_after_conv2-k1+2*p1)/1 + 1
        size_after_conv4 = (size_after_conv3-k1+2*p1)/1 + 1
    
        super(Network5, self).__init__()
        # input image channel, output channels, square convolution kernel
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels = num_layers, out_channels = out_ch_conv1, kernel_size = k1, padding=p1, bias=False)  
        self.conv2 = nn.Conv2d(out_ch_conv1, out_ch_conv2, kernel_size = k1, padding=p1, bias=False)  
        self.conv3 = nn.Conv2d(out_ch_conv2, out_ch_conv3, kernel_size = k1, padding=p1, bias=False)  
        self.conv4 = nn.Conv2d(out_ch_conv3, out_ch_conv4, kernel_size = k1, padding=p1, bias=False)  
        self.fc1 = nn.Linear(int(size_after_conv4**2*out_ch_conv4), 100)
        self.fc2 = nn.Linear(100, num_labels)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)

    def forward(self, x):
        x = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.flatten(x)
        #print('size before fc1:',x.shape)
        x = self.fc1(x)
        #print('size after fc1:',x.shape)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#%%
class Network6(nn.Module):  
    '''conv network used for overfitting/validation of model
    
    validation data: /home/jaspers2/Documents/pixel_optimization/validation_data
    
    '''
    
    def __init__(self, num_pixels_width, num_layers, num_labels, kernel_size=5): 
        
        #kernel/padding values
        k1 = kernel_size       
        k2 = kernel_size*2
        p1 = 1
        
        out_ch_conv1 = 32
        out_ch_conv2 = 32
        size_after_conv1 = (num_pixels_width-k1+2*p1)/1 + 1
        size_after_conv2 = (num_pixels_width-k2+2*p1)/1 + 1
    
        super(Network6, self).__init__()
        # input image channel, output channels, square convolution kernel
        self.flatten = nn.Flatten()
        #self.conv1 = nn.Conv2d(in_channels = num_layers, out_channels = out_ch_conv1, kernel_size = k1, padding=p1, groups=num_layers, bias=False)  
        #self.conv2 = nn.Conv2d(out_ch_conv1, out_ch_conv2, kernel_size = k2, padding=p1, groups=num_layers, bias=False)  
        self.fc1 = nn.Linear(num_layers*num_pixels_width*num_pixels_width, num_layers*num_pixels_width*num_pixels_width)
        self.fc2 = nn.Linear(num_layers*num_pixels_width*num_pixels_width, num_labels)
        
        

    def forward(self, x):
        x = x
        #print('input:',x.shape)
        #x = self.conv1(x)
        #x = F.relu(x)
        #x = self.conv2(x)
        #x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
class Network7(Network):
    
    def __init__(self,input_data, kernel_size=5):
        
        super().__init__(input_data)
        num_pixels_width = input_data.num_pixels_width
        num_channels = input_data.num_channels
        num_labels = len(input_data.labels_names)
        
        
        #kernel/padding values
        k1 = kernel_size       
        k2 = kernel_size*2
        p1 = 1
        
        out_ch_conv1 = 32
        out_ch_conv2 = 32
        size_after_conv1 = (num_pixels_width-k1+2*p1)/1 + 1
        size_after_conv2 = (num_pixels_width-k2+2*p1)/1 + 1
    

        # input image channel, output channels, square convolution kernel
        self.flatten = nn.Flatten()
        #self.conv1 = nn.Conv2d(in_channels = num_layers, out_channels = out_ch_conv1, kernel_size = k1, padding=p1, groups=num_layers, bias=False)  
        #self.conv2 = nn.Conv2d(out_ch_conv1, out_ch_conv2, kernel_size = k2, padding=p1, groups=num_layers, bias=False)  
        self.fc1 = nn.Linear(num_channels*num_pixels_width*num_pixels_width, num_channels*num_pixels_width*num_pixels_width)
        self.fc2 = nn.Linear(num_channels*num_pixels_width*num_pixels_width, num_labels)
        
        

    def forward(self, x):
        x = x
        #print('input:',x.shape)
        #x = self.conv1(x)
        #x = F.relu(x)
        #x = self.conv2(x)
        #x = F.relu(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#%%
class Network8(Network):  
    #fully connected, experimental for random pixel input 10x10
    #SAVE!!!!
    def __init__(self, input_data, kernel_size = 5): #was 512               
        
        super().__init__(input_data)
        
        num_pixels_width = input_data.num_pixels_width
        num_layers = input_data.num_channels
        num_labels = len(input_data.labels_names)
        
        out_ch_conv1 = 10
        k1 = kernel_size   
        p1 = kernel_size
        
        size_after_conv1 = int((num_pixels_width-k1+2*p1)/1 + 1)
        
        #DO NOT EDIT! GOOD PERFORMANCE ON 10X10
        
        # input image channel, output channels, square convolution kernel
        self.conv1 = nn.Conv2d(in_channels = num_layers, out_channels = out_ch_conv1, kernel_size = k1, padding=p1, bias=True) 
        self.flatten = nn.Flatten()
        # an affine operation: y = Wx + b
        # fc implies "fully connected" because linear layers are also called fully connected
        self.fc1 = nn.Linear(out_ch_conv1*size_after_conv1**2, 50)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)
        #self.fc1 = nn.Linear(num_layers*num_pixels_width**2, 100)  
        self.batch_norm1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, num_labels)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)


    def forward(self, x):
        # (1) input layer
        x = x
        #print("shape at input: ",x.shape)
        
        # reshaping before linear layers
        #x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.conv1(x))
        
        x = self.flatten(x)
                
        # () output layer
        #print("shape before fc1: ",x.shape)
        x = F.relu(self.fc1(x)) 
        x = self.batch_norm1(x)
        x = (self.fc2(x))
        #x = self.fc1(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
#%%
class Network9(Network):  
    def __init__(self, input_data, kernel_size = 3): #was 512               
        
        super().__init__(input_data)
        
        num_pixels_width = input_data.num_pixels_width
        num_layers = input_data.num_channels
        num_labels = len(input_data.labels_names)
        
        out_ch_conv1 = 64
        k1 = kernel_size   
        p1 = kernel_size
        s1 = kernel_size
        
        size_after_conv1 = int((num_pixels_width-k1+2*p1)/s1 + 1)
        size_after_conv2 = int((size_after_conv1-k1+2*p1)/s1 + 1)
        size_after_conv3 = int((size_after_conv2-k1+2*p1)/s1 + 1)
        
        # input image channel, output channels, square convolution kernel
        self.conv1 = nn.Conv2d(in_channels = num_layers, out_channels = out_ch_conv1, kernel_size = k1, padding=p1, stride=s1, bias=False) 
        self.conv2 = nn.Conv2d(in_channels = out_ch_conv1, out_channels = out_ch_conv1, kernel_size = k1, padding=p1, stride=s1, bias=False) 
        self.conv3 = nn.Conv2d(in_channels = out_ch_conv1, out_channels = out_ch_conv1, kernel_size = k1, padding=p1, stride=s1, bias=False) 
        self.batch_norm1c = nn.BatchNorm2d(out_ch_conv1)
        self.batch_norm2c = nn.BatchNorm2d(out_ch_conv1)
        self.batch_norm3c = nn.BatchNorm2d(out_ch_conv1)
        self.flatten = nn.Flatten()
        # an affine operation: y = Wx + b
        # fc implies "fully connected" because linear layers are also called fully connected
        self.fc1 = nn.Linear(out_ch_conv1*size_after_conv1**2, 50)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)
        #self.fc1 = nn.Linear(num_layers*num_pixels_width**2, 100)  
        self.batch_norm1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, num_labels)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)


    def forward(self, x):
        # (1) input layer
        x = x
        #print("shape at input: ",x.shape)
        
        # reshaping before linear layers
        #x = x.view(-1, self.num_flat_features(x))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch_norm1c(x)
        
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.batch_norm2c(x)
        
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = self.batch_norm3c(x)
        
        x = self.flatten(x)
                
        # () output layer
        #print("shape before fc1: ",x.shape)
        x = self.fc1(x) 
        x = F.relu(x)
        x = self.batch_norm1(x)
        
        x = (self.fc2(x))
        #x = self.fc1(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features