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
    """normalize images based on given input statistics
    
    :param images: (N,C,H,W) array of images to be normed
    :type images: numpy.ndarray
    
    :param stats: Contains label stastics (max, min, std, mean, image_shape, image_norm_code)
    :type stats: dict
    
    :return: normed_images (Normalized image array)
    :rtype: numpy.ndarray

    """

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


def label_norm_factors(stats, df=torch.empty(1)):
    if type(df) != torch.tensor:
        raise Exception('input to label_norm_factors must be tensor')
    code = stats['label_norm_code']
    

    if code == []:
        stats['denom_factor'] = 1
        stats['centering_factor'] = 1

    if code == 1:
        stats['denom_factor'] = np.array(stats['std']).astype('float32')
        stats['centering_factor'] = np.array(stats['mean']).astype('float32')

    if code == 2:
        stats['denom_factor'] = np.array(stats['max']).astype(
            'float32') - np.array(stats['min']).astype('float32')
        stats['centering_factor'] = np.array(stats['min']).astype('float32')
    
    if code == 3:
        stats['log(Tr_ins)_mean'] = np.log(df['Tr_ins'].mean())
        stats['log(Tr_met)_mean'] = np.log(df['Tr_met'].mean())
        stats['log(Tr_ins)_std'] = np.log(df['Tr_ins'].std())
        stats['log(Tr_met)_std'] = np.log(df['Tr_met'].std())

    return stats

def norm_labels(labels_array, stats):
    """takes torch.Tensor labels array and stats, returns normalized labels.
    
    Must add rescale_labels counterpart as well
    """
    if type(labels_array) != torch.Tensor:
        raise Exception('norm_labels requires torch.Tensor')
    code = stats['label_norm_code']
    names = stats['labels']

    if code == []:
        #none
        raise Exception("missing label norm code")

    if code == 1:
        #mean/stdev (astype 'float32'?)
        denom_factor = torch.tensor(stats['std'])
        centering_factor = torch.tensor(stats['mean'])
        normed_labels = (labels_array - centering_factor)/denom_factor
        
    if code == 2:
        #max-min
        denom_factor = torch.tensor(stats['max']) - torch.tensor(stats['min'])
        centering_factor = torch.tensor(stats['min'])
        normed_labels = (labels_array - centering_factor)/denom_factor
    
    if code == 3:
        normed_labels = torch.empty(labels_array.shape)
        
        i1 = names.index("Tr_ins")
        i2 = names.index("Tr_met")
        normed_labels[:,i1] = (torch.log(1/labels_array[:,i1]))
        normed_labels[:,i2] = (torch.log(1/labels_array[:,i2]))
        
        i3 = names.index("Temp")
        denom_factor = torch.tensor(stats['max']) - torch.tensor(stats['min'])
        centering_factor = torch.tensor(stats['min'])
        normed_labels[:,i3] = (labels_array[:,i3]-centering_factor[i3])/denom_factor[i3]      
    
    return normed_labels

def rescale_labels(normed_labels, stats):
    """
    """
    if type(normed_labels) != torch.Tensor:
        raise Exception('rescale_labels requires torch.Tensor')
    code = stats['label_norm_code']
    names = stats['labels']
    
    if code == []:
        rescaled_labels = normed_labels
    
    if code == 1:
        #mean/stdev
        denom_factor = torch.tensor(stats['std'])
        centering_factor = torch.tensor(stats['mean'])
        rescaled_labels = normed_labels*denom_factor + centering_factor
        
    if code == 2:
        #max-min
        denom_factor = torch.tensor(stats['max']) - torch.tensor(stats['min'])
        centering_factor = torch.tensor(stats['min'])
        rescaled_labels = normed_labels*denom_factor + centering_factor
    
    if code == 3:
        rescaled_labels = torch.empty(normed_labels.shape)
        i1 = names.index("Tr_ins")
        i2 = names.index("Tr_met")
        rescaled_labels[:,i1] = (torch.exp(-normed_labels[:,i1]))
        rescaled_labels[:,i2] = (torch.exp(-normed_labels[:,i2]))
        
        i3 = names.index("Temp")
        denom_factor = torch.tensor(stats['max']) - torch.tensor(stats['min'])
        centering_factor = torch.tensor(stats['min'])
        rescaled_labels[:,i3] = normed_labels[:,i3]*denom_factor[i3] + centering_factor[i3]
    
    return rescaled_labels


class Network(nn.Module):
    
    def __init__(self, input_data):
        super(Network, self).__init__()
        
        self.image_stats = input_data.image_stats
        self.label_stats = input_data.label_stats
        #self.image_centering_factor = input_data.image_centering_factor
        #self.image_denom_factor = input_data.image_denom_factor
        #self.label_centering_factor = input_data.label_centering_factor
        #self.label_denom_factor = input_data.label_denom_factor
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
        #self.conv2 = nn.Conv2d(in_channels = out_ch_conv1, out_channels = out_ch_conv1, kernel_size = k1, padding=p1, stride=s1, bias=False) 
        #self.conv3 = nn.Conv2d(in_channels = out_ch_conv1, out_channels = out_ch_conv1, kernel_size = k1, padding=p1, stride=s1, bias=False) 
        self.batch_norm1c = nn.BatchNorm2d(out_ch_conv1)
        #self.batch_norm2c = nn.BatchNorm2d(out_ch_conv1)
        #self.batch_norm3c = nn.BatchNorm2d(out_ch_conv1)
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
    
#%%
class Network10(Network):  
    def __init__(self, input_data, out_chnl = [10], kernel_size = [3], stride_size = [3], padding_size = [3]): #was 512               
        """
        """
        super().__init__(input_data)
        
        num_pixels_width = input_data.num_pixels_width
        num_layers = input_data.num_channels
        num_labels = len(input_data.labels_names)
        
        # out_ch_conv1 = 10
        # out_ch_conv2 = 5
        # out_ch_conv3 = 5
        # out_ch_conv4 = 5
        # k1 = kernel_size[0]
        # p1 = kernel_size[0]
        # s1 = 3#kernel_size
        
        # size_after_conv1 = int((num_pixels_width-k1+2*p1)/s1 + 1)
        # size_after_conv2 = int((size_after_conv1-k1+2*p1)/s1 + 1)
        # size_after_conv3 = int((size_after_conv2-k1+2*p1)/s1 + 1)
        # size_after_conv4 = int((size_after_conv3-k1+2*p1)/s1 + 1)
        
        self.conv_layers = nn.ModuleList()
        #self.batchnorm_layers = nn.ModuleList()
        #self.pool_layers = nn.ModuleList()
        prev_layer_size = num_layers
        out_size = num_pixels_width
        for i in range(len(out_chnl)):
            next_layer = nn.Conv2d(prev_layer_size,
                                   out_chnl[i],
                                   kernel_size[i],
                                   stride_size[i],
                                   padding_size[i],
                                   bias=False)
            self.conv_layers.append(next_layer)
            #pool_layer = nn.MaxPool2d(2,stride=1)
            #self.pool_layers.append(pool_layer)
            
            #batch_layer = nn.BatchNorm2d(out_chnl[i])
            #self.batchnorm_layers.append(batch_layer) #commenting out, doesn't seem to help
            conv_out_size = int((out_size-kernel_size[i]+2*padding_size[i])/stride_size[i] + 1)
            #pool_out_size = int((conv_out_size+2*0-1*(2-1)-1)/1 + 1)
            print(f"layer {i} output size: {conv_out_size}")
            prev_layer_size = out_chnl[i]
            out_size = conv_out_size
        
        # input image channel, output channels, square convolution kernel
        # self.conv1 = nn.Conv2d(in_channels = num_layers, out_channels = out_ch_conv1, kernel_size = k1, padding=p1, stride=s1, bias=False) 
        # self.batch_norm1c = nn.BatchNorm2d(out_ch_conv1)
        
        # self.conv2 = nn.Conv2d(in_channels = out_ch_conv1, out_channels = out_ch_conv2, kernel_size = k1, padding=p1, stride=s1, bias=False) 
        # self.batch_norm2c = nn.BatchNorm2d(out_ch_conv2)
        
        # self.conv3 = nn.Conv2d(in_channels = out_ch_conv2, out_channels = out_ch_conv3, kernel_size = k1, padding=p1, stride=s1, bias=False) 
        # self.batch_norm3c = nn.BatchNorm2d(out_ch_conv3)
        
        # self.conv4 = nn.Conv2d(in_channels = out_ch_conv3, out_channels = out_ch_conv4, kernel_size = k1, padding=p1, stride=s1, bias=False) 
        # self.batch_norm4c = nn.BatchNorm2d(out_ch_conv4)
        
        self.flatten = nn.Flatten()
        # an affine operation: y = Wx + b
        # fc implies "fully connected" because linear layers are also called fully connected
        #self.fc1 = nn.Linear(out_ch_conv1*size_after_conv1**2, 50)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)
        #self.batch_norm1 = nn.BatchNorm1d(50)
        self.fc1 = nn.Linear(out_chnl[-1]*out_size**2, num_labels)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)
        
        
        #dropout
        self.drop_layer1 = nn.Dropout(0) 
        self.drop_layer2 = nn.Dropout(0)
        self.drop_layer3 = nn.Dropout(0) #was 0.02
        self.drop_layer4 = nn.Dropout(0) #was 0.02
        self.drop_layer5 = nn.Dropout(0) #was 0.02
        
        #initialization
        # nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        
        for i,layer in enumerate(self.conv_layers):
            x = F.relu(layer(x))
            #x = self.pool_layers[i](x)
            #x = self.batchnorm_layers[i](x)
        
        # x = self.conv1(x)
        # x = self.drop_layer1(F.relu(x))
        # x = self.batch_norm1c(x)
        
        # x = self.conv2(x)
        # x = self.drop_layer2(F.relu(x))
        # x = self.batch_norm2c(x)
        
        # x = self.conv3(x)
        # x = self.drop_layer3(F.relu(x))
        # x = self.batch_norm3c(x)
        
        # x = self.conv4(x)
        # x = self.drop_layer4(F.relu(x))
        # x = self.batch_norm4c(x)
        
        x = self.flatten(x)
                
        # () output layer
        #print("shape before fc1: ",x.shape)
        x = self.fc1(x) 
        #x = self.drop_layer5(F.relu(x))
        #x = self.batch_norm1(x)
        
        #x = (self.fc2(x))
        #x = self.fc1(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features