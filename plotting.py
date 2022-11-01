#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 09:04:05 2022

@author: jaspers2
"""

import numpy as np
import pandas as pd
import os
import image_creation
import matplotlib.pyplot as plt
import pixel_optim_nn
import perf_net
import pixel_nn
import torch

plt.rcParams["figure.dpi"] = 500

results_folder = input("combined_results folder: ")
trained_model_folder = input("trained model folder: ")
perfnn = pixel_optim_nn.load_perfnet(trained_model_folder)

dataloader = torch.load(trained_model_folder+'/test_dataloader.pkl')

evaluate = pixel_nn.Evaluate(dataloader, perfnn)
evaluate.get_preds('cpu')
evaluate.pred_report()
evaluate.plot_results()

###############################################
labels = ['Extinction Ratio', 'Temperature Rise']

sub = evaluate.predictions
#sub = evaluate.predictions[evaluate.predictions[:,0]>8.96]
#sub = sub[sub[:,0]< 12.14]
#train = sub[sub[:,1] == max(sub[:,1])]
#ext_error = max(sub[:,0])-min(sub[:,0])


ext_comsol_data = {'1':12.14,
                   '2': 8.96, 
                   '3': 11.44,
                   'CB': 12.39}#, 
                   #'Train':sub[:,0].tolist()}

ext_nn_data = {'1':10.01,
               '2':10.22,
               '3':10.03}

ext_comsol_label = {'1': 12.5,
                    '2': 7.5,
                    '3': 12.0,
                    'CB': 11.0}

ext_nn_label = {'1': 8.0,
                '2': 11.0,
                '3': 8}

temp_comsol_data = {'1':13.77,
                    '2':11.31,
                    '3':13.86,
                    'CB': 2.45}#,
                    #'Train':sub[:,1].tolist()}

temp_nn_data = {'1':10.94,
                '2':11.49,
                '3':11.83}

temp_comsol_label = {'1':12.25,
                    '2':10.0,
                    '3':12.5,
                    'CB': 3.5}

temp_nn_label = {'1':9,
                '2':12.5,
                '3':10.1}

# design1 = [12.14,13.77]
# design2 = [8.96,11.31]
# design3 = [11.44,13.86]
# train = [train[0][0],train[0][1]]

fig,axis = plt.subplots(1,2,figsize = (9,3))
axis[0].scatter(ext_nn_data.keys(),ext_nn_data.values(),s=75,marker='X',label='NN')
axis[0].scatter(ext_comsol_data.keys(),ext_comsol_data.values(),marker='o',label='COMSOL',color='orange')
axis[0].axhline(10,linestyle = 'dashed', label = 'target value')
#axis[0].set_ylim(top=12.75)
loc = []
for i in range(len(sub[:,0])):
    loc.append('Train')
axis[0].scatter(loc,sub[:,0],marker='.',color='orange')
axis[0].set_ylabel('Extinction Ratio (dB)')
axis[0].set()
fig.legend(loc='lower center',ncol = len(ext_comsol_data), bbox_to_anchor=(0.5,-.1))
axis[0].grid()

for i in range(3):
    axis[0].text(i,ext_comsol_label[str(i+1)],str(ext_comsol_data[str(i+1)]))
    axis[0].text(i,ext_nn_label[str(i+1)],str(ext_nn_data[str(i+1)]))

axis[0].text(3,ext_comsol_label['CB'],str(ext_comsol_data['CB']))

axis[1].scatter(temp_nn_data.keys(),temp_nn_data.values(),s=75,marker='X',label='NN')
axis[1].scatter(temp_comsol_data.keys(),temp_comsol_data.values(),marker='o',label='COMSOL',color='orange')
axis[1].scatter(loc,sub[:,1],marker='.',color='orange')
for i in range(3):
    axis[1].text(i,temp_comsol_label[str(i+1)],str(temp_comsol_data[str(i+1)]))
    axis[1].text(i,temp_nn_label[str(i+1)],str(temp_nn_data[str(i+1)]))

axis[1].text(3,temp_comsol_label['CB'],str(temp_comsol_data['CB']))

axis[1].set_ylabel('Temperature Rise (K)')
axis[1].grid()

fig.tight_layout()