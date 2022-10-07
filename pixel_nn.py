#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:01:38 2021

@author: jaspers2
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import datetime
import os
import pickle
from sklearn.model_selection import train_test_split
import perf_net


class InputData():

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_import_pixel()

    def data_import_pixel(self):
        print("import data")
        
        self.orig_images = np.load(
            self.data_dir + '/feature_images.npy').astype(np.float32)
    
        result_labels_orig = pd.read_csv(
            self.data_dir + '/final_comsol_results.csv', delimiter=',')

        self.orig_labels = result_labels_orig.astype('float32').values
        self.labels_names = result_labels_orig.columns.to_list()

        N, C, H, W = self.orig_images.shape
        self.num_channels = C
        self.num_pixels = H*W
        self.num_pixels_width = W

        print('import done')


    def create_datasets(self, test_perc, n_batch, image_norm_code=[], label_norm_code=[]):
        # split dataset into train and test
        images_train, images_test, labels_train, labels_test = train_test_split(
            self.orig_images, self.orig_labels, test_size=test_perc, shuffle=True)

        # image stats, only use training images for stats
        self.image_stats = self.get_image_stats(images_train)  
        self.image_stats['image_shape'] = self.orig_images.shape
        self.image_stats['image_norm_code'] = image_norm_code

        # label stats, only use training images for stats
        self.label_stats = self.get_labels_stats(labels_train)  
        self.label_stats['label_norm_code'] = label_norm_code
        self.label_stats['labels'] = self.labels_names
        
        self.train_dataloader = create_dataloader(self.image_stats, 
                                                  self.label_stats, 
                                                  images_train, 
                                                  labels_train, 
                                                  n_batch)
        self.test_dataloader = create_dataloader(self.image_stats, 
                                                 self.label_stats, 
                                                 images_test, 
                                                 labels_test, 
                                                 n_batch)

    
    def get_labels_stats(self, array):
        array_max = array.max(axis=0)
        array_min = array.min(axis=0)
        array_std = array.std(axis=0)
        array_mean = array.mean(axis=0)

        stats = {'max': array_max.tolist(),
                 'min': array_min.tolist(),
                 'std': array_std.tolist(),
                 'mean': array_mean.tolist()}

        return stats

    def get_image_stats(self, data):
        N, C, H, W = data.shape
        max_val = []
        min_val = []
        mean_val = []
        std_val = []

        for i in range(C):
            max_val += [data[:, i, :, :].max()]
            min_val += [data[:, i, :, :].min()]
            mean_val += [data[:, i, :, :].mean()]
            std_val += [data[:, i, :, :].std()]

        stats = {'max': max_val, 'min': min_val,
                 'std': std_val, 'mean': mean_val}

        return stats


class Evaluate():
    """
    """
    def __init__(self, eval_dataloader, network):
        """
        """
        self.eval_dataloader = eval_dataloader
        self.network = network

    def get_preds(self, device):

        with torch.no_grad():
            preds_all, error_all, labels_all = get_all_preds(
                self.eval_dataloader, self.network, device)

        preds_all_rescaled = perf_net.rescale_labels(preds_all, self.network.label_stats)
        preds_all_rescaled = preds_all_rescaled.cpu().numpy()

        labels_all_rescaled = perf_net.rescale_labels(labels_all,self.network.label_stats)
        labels_all_rescaled = labels_all_rescaled.cpu().numpy()
                                                      
        self.predictions = preds_all_rescaled
        self.actual_values = labels_all_rescaled

    def pred_report(self):
        self.diff = abs(self.predictions - self.actual_values)
        self.error = abs(self.diff)/self.actual_values

        print('labels:', self.network.label_names)
        print('average abs(preds-labels):', self.diff.mean(axis=0))
        print('max abs(preds-labels):', self.diff.max(axis=0))
        print('average abs error [%]:', self.error.mean(axis=0)*100)
        print('\n')

    def plot_results(self):

        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['font.size'] = 14     
        
        fig, ax = plt.subplots(1,len(self.network.label_names),figsize=(10,5))
        fig.tight_layout(h_pad=6)
        for i in range(len(self.network.label_names)):
            axis_max = max(np.array([self.predictions[:,i],self.actual_values[:,i]]).reshape(-1,1))
            ax[i].axis('equal')
            ax[i].scatter(self.actual_values[:, i], self.predictions[:, i],s=5)
            ax[i].set(xlim=(0,axis_max),ylim=(0,axis_max),xlabel='Actual values')
            if i == 0:
                ax[i].set_ylabel('Predictions')
            
            title = self.network.label_names[i]
            if title == "ext_ratio":
                title = "Extinction Ratio"
            if title == "Temp":
                title = "Temperature"
            if title == "dT":
                title = "Temperature Rise"
            ax[i].set_title(title)
            ax[i].grid()

        
        for i in range(len(self.network.label_names)):
            plt.figure()
            plt.hist(self.error[:,i]*100)
            plt.xlabel("Percent error")
            plt.title(self.network.label_names[i])


#########################################################
def create_dataloader(image_stats,label_stats,images,labels,n_batch):
    """Create dataloader from set of images and labels

    Parameters
    ----------
    image_stats : dict
        Dictionary of image statistics
    label_stats : dict
        Dictionary of label statistics
    images : np.array
        Image arrays, N,C,H,W 
    labels : np.array
        Array of labels
    n_batch : int
        Batch size for dataloader (to be used during training?)

    Returns
    -------
    dataloader : torch DataLoader
        DataLoader instance created

    """
    # normalize images
    images_normed = perf_net.norm_images(
        images, image_stats).astype('float32')
    
    images_normed_tf = torch.tensor(images_normed)

    # normalize responses
    labels_tf = torch.tensor(labels)
    labels_normed_tf = perf_net.norm_labels(labels_tf,label_stats)

    dataset = TensorDataset(images_normed_tf, labels_normed_tf)

    dataloader = DataLoader(dataset, batch_size=n_batch)
    return dataloader

def train(dataloader, model, loss_fn, optimizer, device, train_error):
    """
    """

    model.train()
    train_loss = 0
    batch_num = 0
    for batch, (X, y) in enumerate(dataloader):
        batch_num += 1
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= batch_num

    train_error.append(train_loss)

    return train_error

#########################################################
def test(dataloader, model, loss_fn, device, test_error, error_flag=False):
    """
    """

    model.eval()
    test_loss = 0
    error_calc = []
    batch_num = 0
    with torch.no_grad():
        for X, y in dataloader:
            batch_num += 1
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= batch_num

    test_error.append(test_loss)

    if error_flag == True:
        return(error_calc)
    else:
        return(test_error)


#########################################################
def get_all_preds(dataloader, model, device):
    all_labels = torch.tensor([]).to(device)
    all_preds = torch.tensor([]).to(device)
    all_error = torch.tensor([]).to(device)

    model.eval()

    for batch in dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        all_labels = torch.cat((all_labels, labels), dim=0)

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

        error = 100*(preds - labels)/labels
        all_error = torch.cat((all_error, error), dim=0)

    return all_preds, all_error, all_labels
#########################################################


def save_model(input_data, network, data_directory):

    # make timestamp
    date = datetime.datetime.now()
    timestamp = (str(date.year)[-2:] + str(date.month).rjust(2, '0') +
                 str(date.day).rjust(2, '0')
                 + '-' + str(date.hour).rjust(2, '0') +
                 str(date.minute).rjust(2, '0'))

    # create directory
    new_directory = os.path.join(data_directory, 'trained_model_'+timestamp)
    os.mkdir(new_directory)
    print("directory created: ", new_directory)

    # save model
    torch.save(network.state_dict(), new_directory + '/trained_model.pth')

    with open(new_directory + '/network.pkl', 'wb') as outp:
        pickle.dump(network, outp, pickle.HIGHEST_PROTOCOL)

    # save input data (denotes which is training/test; useful for future NN eval)
    torch.save(input_data.train_dataloader,new_directory + '/train_dataloader.pkl')
    torch.save(input_data.test_dataloader,new_directory + '/test_dataloader.pkl')
    
    # save image stats
    image_dict = {}
    image_dict.update(input_data.image_stats)
    with open(os.path.join(new_directory, 'image_stats.txt'), 'w') as output_file:
        output_file.write(str(image_dict))

    # save label state
    label_dict = {}
    label_dict.update(input_data.label_stats)
    with open(os.path.join(new_directory, 'label_stats.txt'), 'w') as output_file:
        output_file.write(str(label_dict))

    return
#########################################################


# %%
def main(
    #num_epochs = 500,
    num_epochs = 300, #temp change for dummy data
    learning_rate = 0.001,
    out_chnl = [30,10,10,10],
    kernel_size = [3,3,3,3],
    stride_size = [1,2,2,2],
    padding_size = [1,1,1,1],
    n_batch_in = 2**8 #was 2**3
    ):
    """
    """
    #######################
    # initial inputs
    data_dir = input('paste data directory:  ')
    network_name = input('network name:   ')
    network_to_use = getattr(perf_net, network_name)
    use_gpu = False  # manual override for gpu option; having issues with pixel_optim_nn on gpu

    
    if use_gpu == True:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

    print("Using {} device".format(device))

    # turn off/on comutation graph
    torch.set_grad_enabled(True)  # debug 220218 turned off

    input_data = InputData(data_dir)
    input_data.create_datasets(test_perc=0.2,
                               n_batch=n_batch_in,
                               image_norm_code=1,
                               label_norm_code=2)

    network = network_to_use(input_data, out_chnl, kernel_size, stride_size, padding_size).to(device)
    network.network_name = network_name

    # output parameters of model
    params = list(network.parameters())
    print("length of parameters = ", len(params))
    # print("conv1's weight: ",params[0].size())  # conv1's .weight
    print('Trainable parameters:', sum(p.numel()
          for p in network.parameters() if p.requires_grad))
    print('---------------------------')
    
    optimizer_in = optim.Adam(network.parameters(),
                              lr=learning_rate)  # from deeplizard
    #momentum = 0.87
    #optimizer_in = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum) #from pytorch tutorial
    loss_fn_in = nn.MSELoss()  # MSE for continuous label
    test_error = []
    train_error = []

    for t in range(num_epochs):
        train_error = train(input_data.train_dataloader, network,
                            loss_fn_in, optimizer_in, device, train_error)
        test_error = test(input_data.test_dataloader, network,
                          loss_fn_in, device, test_error, error_flag=False)
        if (t+1)%10 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"Avg training loss: {train_error[-1]:>7f}")
            print(f"Avg test loss: {test_error[-1]:>8f} \n")
    print("Done!")

    ########################################################
    # plot test/train error
    plt.plot(np.array(test_error), label='test_error')
    plt.plot(np.array(train_error), label='train_error')
    plt.legend()

    ########################################################
    # get predictions for all data

    print('----Evaluate test data----')
    evaluate = Evaluate(input_data.test_dataloader, network)
    evaluate.get_preds(device)
    evaluate.pred_report()
    evaluate.plot_results()

    print('----Evaluate training data----')
    evaluate_train = Evaluate(input_data.train_dataloader, network)
    evaluate_train.get_preds(device)
    evaluate_train.pred_report()
    evaluate_train.plot_results()

    ###########################################
    if input('save NN model + stats? y to save:    ') == 'y':
        save_model(input_data, network, data_dir)

    return


if __name__ == '__main__':
    main()