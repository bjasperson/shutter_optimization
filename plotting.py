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

plt.rcParams["figure.dpi"] = 500


#def plot_perfnn_performance():
trained_model_folder = input("trained model folder: ")
results_folder = input("combined_results folder: ")
perfnn = pixel_optim_nn.load_perfnet(trained_model_folder)
df_results = pd.read_pickle(results_folder+"/df_all.pkl")
#need to normalize input data based on perfnn stats
images = df_results['image'].to_numpy()
images = np.array([x for x in images])
images = images[:,:,:10,:10]
labels = df_results[['ext_ratio','T_VO2_avg']].to_numpy()
dataloader = pixel_nn.create_dataloader(perfnn.image_stats, 
                                        perfnn.label_stats, 
                                        images, 
                                        labels, 
                                        2**8)


evaluate_train = pixel_nn.Evaluate(dataloader, perfnn)
evaluate_train.get_preds('cpu')
evaluate_train.pred_report()
evaluate_train.plot_results()
    
    #return perfnn, df_results
