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


#def plot_perfnn_performance():
trained_model_folder = input("trained model folder: ")
results_folder = input("combined_results folder: ")
perfnn = pixel_optim_nn.load_perfnet(trained_model_folder)
df_results = pd.read_pickle(results_folder+"/df_all.pkl")
#need to normalize input data based on perfnn stats
images = df_results['image'].to_numpy()
images = np.array([x for x in images])
images_normed = perf_net.norm_images(images, perfnn.image_stats)
DataLoader(train_dataset, batch_size=n_batch)
#create dataloader
evaluate_train = pixel_nn.Evaluate(input_data.train_dataloader, perfnn)
evaluate_train.get_preds(device)
evaluate_train.pred_report()
evaluate_train.plot_results()
    
    #return perfnn, df_results
