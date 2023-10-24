#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting functionality for journal paper
"""


import matplotlib.pyplot as plt
import pixel_optim_nn
import data_merge
import pixel_nn
import torch
import os

plt.rcParams["figure.dpi"] = 500

results_folder = "./data/combined_results_dT"
trained_model_folder = results_folder + "/trained_model_221004-1433"
perfnn = pixel_optim_nn.load_perfnet(trained_model_folder)

dataloader_train = torch.load(trained_model_folder+'/train_dataloader.pkl')
dataloader_test = torch.load(trained_model_folder+'/test_dataloader.pkl')

data_merge.analyze_results(results_folder, 'y')  


# test evaluations
evaluate = pixel_nn.Evaluate(dataloader_test, perfnn, "./figs")
evaluate.get_preds('cpu')
evaluate.pred_report()
evaluate.plot_results(save='y') 
evaluate.plot_residual()
evaluate.add_perc_coverage('cpu')

evaluate_train = pixel_nn.Evaluate(dataloader_train, perfnn, "./figs")
evaluate_train.get_preds('cpu')
evaluate_train.add_perc_coverage('cpu')

###############################################

def plot_data_distribution(data, fig, axs):
    loc_er = data.network.label_names.index('ext_ratio')
    loc_dT = data.network.label_names.index('dT')
    ext_ratio = data.actual_values[:,loc_er]
    temp = data.actual_values[:,loc_dT]
    perc_cov = data.perc_cov

    #fig.tight_layout(h_pad=6)
    axs[0].hist(ext_ratio)
    axs[1].hist(temp)
    axs[2].hist(perc_cov)
    axs[0].set_xlabel('Ext. Ratio')
    axs[1].set_xlabel('dT')
    axs[2].set_xlabel('Percent Coverage')
    fig.supylabel('Count')
    return fig, axs


def fig5b(train, test, save = 'y'):
    fig, axs = plt.subplots(1,3,figsize = (12,5),sharey=True)
    fig, axs  = plot_data_distribution(train, fig, axs)
    fig, axs  = plot_data_distribution(test, fig, axs)
    fig.legend(['train','test'], 
               loc = 'upper right',
               #bbox_to_anchor = (0.5,-0.05),
               ncol = 2)
    fig.tight_layout()
    if save == 'y':
        fig.savefig('./figs/FIG5b_data_hist.eps')
    else:
        plt.show()
    return

def fig8():
    sub = evaluate.predictions
    ext_comsol_data = {'1':12.14,
                    '2': 8.96, 
                    '3': 11.44,
                    'CB': 12.39} 

    ext_nn_data = {'1':10.01,
                '2':10.22,
                '3':10.03}

    temp_comsol_data = {'1':13.77,
                        '2':11.31,
                        '3':13.86,
                        'CB': 2.45}

    temp_nn_data = {'1':10.94,
                    '2':11.49,
                    '3':11.83}

    fig,axis = plt.subplots(1,2,figsize = (9,3))
    axis[0].boxplot(sub[:,0],positions=[4],labels=['Train'])
    axis[0].scatter(ext_nn_data.keys(),ext_nn_data.values(),s=75,marker='X',label='NN')
    axis[0].scatter(ext_comsol_data.keys(),ext_comsol_data.values(),marker='o',label='FEM')
    axis[0].axhline(10,linestyle = 'dashed', label = 'target value')
    loc = []
    for i in range(len(sub[:,0])):
        loc.append('Train')

    axis[0].set_ylabel('Extinction Ratio (dB)')
    axis[0].set()
    fig.legend(loc='lower center',ncol = len(ext_comsol_data), bbox_to_anchor=(0.5,-.1))
    axis[0].grid()
    axis[0].set_xticks([0,1,2,3,4],['1','2','3','CB','Train'])


    axis[1].boxplot(sub[:,1],positions=[4],labels=['Train'])
    axis[1].scatter(temp_nn_data.keys(),temp_nn_data.values(),s=75,marker='X',label='NN')
    axis[1].scatter(temp_comsol_data.keys(),temp_comsol_data.values(),marker='o',label='FEM')

    axis[1].set_ylabel('Temperature Rise (K)')
    axis[1].grid()
    axis[1].set_xticks([0,1,2,3,4],['1','2','3','CB','Train'])

    fig.tight_layout()
    fig.savefig(os.path.expanduser("./figs/FIG8_opt_design_data.eps"),bbox_inches = "tight")
    fig.savefig(os.path.expanduser("./figs/FIG8_opt_design_data.png"),bbox_inches = "tight")


def main():
    fig5b(evaluate_train, evaluate)
    fig8()

if __name__ == "__main__":
    main()