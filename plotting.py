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

