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
    """
    normalize images based on given input statistics
    
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
        self.label_names = input_data.labels_names
    
class Network_PerfNet(Network):  
    def __init__(self, input_data, out_chnl = [10], kernel_size = [3], stride_size = [3], padding_size = [3]): #was 512               
        """
        """
        super().__init__(input_data)
        
        num_pixels_width = input_data.num_pixels_width
        num_layers = input_data.num_channels
        num_labels = len(input_data.labels_names)
        
        self.conv_layers = nn.ModuleList()
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

            conv_out_size = int((out_size-kernel_size[i]+2*padding_size[i])/stride_size[i] + 1)
            print(f"layer {i} output size: {conv_out_size}")
            prev_layer_size = out_chnl[i]
            out_size = conv_out_size
        
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(out_chnl[-1]*out_size**2, num_labels)  #last 3x dims going into reshaping (was 96*496*496 for 500 pixel)        
        
        #dropout
        self.drop_layer1 = nn.Dropout(0) 
        self.drop_layer2 = nn.Dropout(0)
        self.drop_layer3 = nn.Dropout(0) #was 0.02
        self.drop_layer4 = nn.Dropout(0) #was 0.02
        self.drop_layer5 = nn.Dropout(0) #was 0.02
        
    def forward(self, x):
        
        for i,layer in enumerate(self.conv_layers):
            x = F.relu(layer(x))
        x = self.flatten(x)
        x = self.fc1(x) 
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features