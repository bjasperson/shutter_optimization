#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:01:38 2021

@author: jaspers2
"""

# 220101: original file in python:sandia:cnn_optimization
################################################
# TODO:
# save NN along with data stats for optim_nn
# need stats to convert final (normalized) image to actual size
################################################

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


#from ray import tune
#from ray.tune.schedulers import ASHAScheduler


# plan:
# use pickle to capture class at runtime, save in folder
# capture the norm method used along with net structure
# roll perf_net into this file.
# no need to load perf_net class in top_opt. just use
# can even save the stats in the class instance! so no text file needed (although may want human readable)
# https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence


# %%
#########################################################
class InputData():

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_import_pixel()

    def data_import_pixel(self):
        print("import data")

        # ------------------
        # import image file (.py) and results file (.txt)
        # images
        
        self.orig_images = np.load(
            self.data_dir + '/feature_images.npy').astype(np.float32)
    
                

        # ------------------
        # labels:
        result_labels_orig = pd.read_csv(
            self.data_dir + '/final_comsol_results.csv', delimiter=',')
        # result_labels_orig.set_index(index_labels,inplace=True)

        # ------------------
        # image key to match order of images to labels
        # image_key = pd.read_csv(data_directory + '/key.csv',index_col=(0))
        # self.image_key = image_key/10**9

        # ------------------
        # sort labels based on image key
        # orig_labels_raw = result_labels_orig.astype('float32')
        # labels_ordered = pd.merge(self.image_key,orig_labels_raw, how='left', on=index_labels)
        # labels_ordered.drop(index_labels,axis=1,inplace=True)
        self.orig_labels = result_labels_orig.astype('float32').values
        self.labels_names = result_labels_orig.columns.to_list()

        N, C, H, W = self.orig_images.shape
        self.num_channels = C
        self.num_pixels = H*W
        self.num_pixels_width = W

        # ------------------
        print('import done')

    def create_datasets(self, test_perc, n_batch, image_norm_code=[], label_norm_code=[]):

        # split dataset into train and test
        images_train, images_test, labels_train, labels_test = train_test_split(
            self.orig_images, self.orig_labels, test_size=test_perc, shuffle=True)

        #################################
        # normalize images
        self.image_stats = self.get_image_stats(
            images_train)  # only use training images for stats
        self.image_stats['image_shape'] = self.orig_images.shape
        self.image_stats['image_norm_code'] = image_norm_code

        images_train_normed = perf_net.norm_images(
            images_train, self.image_stats).astype('float32')
        images_test_normed = perf_net.norm_images(
            images_test, self.image_stats).astype('float32')

        # assign to torch tensor
        images_train_tf = torch.tensor(images_train_normed)
        images_test_tf = torch.tensor(images_test_normed)

        #################################
        # normalize responses
        self.label_stats = self.get_labels_stats(
            labels_train)  # only use training images for stats
        self.label_stats['label_norm_code'] = label_norm_code
        self.label_stats['labels'] = self.labels_names

        label_centering_factor, label_denom_factor = self.label_norm_factors()
        self.label_centering_factor = label_centering_factor
        self.label_denom_factor = label_denom_factor

        labels_train_normed = perf_net.norm_labels(
            labels_train, label_centering_factor, label_denom_factor)
        labels_test_normed = perf_net.norm_labels(
            labels_test, label_centering_factor, label_denom_factor)

        labels_train_tf = torch.tensor(labels_train_normed)
        labels_test_tf = torch.tensor(labels_test_normed)

        train_dataset = TensorDataset(images_train_tf, labels_train_tf)
        test_dataset = TensorDataset(images_test_tf, labels_test_tf)

        #################################
        # create dataloaders
        self.train_dataloader = DataLoader(train_dataset, batch_size=n_batch)
        self.test_dataloader = DataLoader(test_dataset, batch_size=n_batch)

    def get_labels_stats(self, array):
        array_max = array.max(axis=0)
        array_min = array.min(axis=0)
        #df_orig_avg = (df_orig_max+df_orig_min)/2
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

    def label_norm_factors(self):
        stats = self.label_stats
        code = stats['label_norm_code']

        if code == []:
            denom_factor = 1
            centering_factor = 1

        if code == 1:
            denom_factor = np.array(stats['std']).astype('float32')
            centering_factor = np.array(stats['mean']).astype('float32')
            pass

        if code == 2:
            denom_factor = np.array(stats['max']).astype(
                'float32') - np.array(stats['min']).astype('float32')
            centering_factor = np.array(stats['min']).astype('float32')
            pass

        return centering_factor, denom_factor


# %%
class Evaluate():

    def __init__(self, eval_dataloader, network):
        self.eval_dataloader = eval_dataloader
        self.network = network

    def get_preds(self, device):

        with torch.no_grad():
            preds_all, error_all, labels_all = get_all_preds(
                self.eval_dataloader, self.network, device)

        preds_all = preds_all.cpu().numpy()
        labels_all = labels_all.cpu().numpy()

        preds_all_rescaled = perf_net.rescale_labels(preds_all,
                                                     self.network.label_centering_factor,
                                                     self.network.label_denom_factor)

        labels_all_rescaled = perf_net.rescale_labels(labels_all,
                                                      self.network.label_centering_factor,
                                                      self.network.label_denom_factor)

        self.predictions = preds_all_rescaled
        self.actual_values = labels_all_rescaled

    def pred_report(self):
        self.diff = abs(self.predictions - self.actual_values)
        self.error = self.diff/self.actual_values

        print('labels:', self.network.label_names)
        print('average abs(preds-labels):', self.diff.mean(axis=0))
        print('max abs(preds-labels):', self.diff.max(axis=0))
        print('min abs(preds-labels):', self.diff.min(axis=0))
        print('average error [%]:', self.error.mean(axis=0)*100)
        print('\n')

    def plot_results(self):

        for i in range(len(self.network.label_names)):
            plt.figure()
            plt.scatter(self.actual_values[:, i], self.predictions[:, i])
            plt.xlabel('Actual values')
            plt.ylabel('Predictions')
            plt.title(self.network.label_names[i])
            plt.grid()


#########################################################
def train(dataloader, model, loss_fn, optimizer, device, train_error):
    #size = len(dataloader.dataset)
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

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= batch_num

    train_error.append(train_loss)
    print(f"Avg training loss: {train_loss:>7f}")

    return train_error

#########################################################


def test(dataloader, model, loss_fn, device, test_error, error_flag=False):
    #size = len(dataloader.dataset)
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

            # problematic when y=0, which happens w/ normalized data
            # if error_flag == True:
            #     batch_error = 100*(pred-y)/y
            #     print(batch_error)
            #     batch_error_list = batch_error.tolist()
            #     error_calc.append(batch_error_list)
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= batch_num

    test_error.append(test_loss)
    # correct /= size #legacy code from pytorch
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Avg test loss: {test_loss:>8f} \n")

    if error_flag == True:
        return(error_calc)
    else:
        return(test_error)


#########################################################
def get_all_preds(dataloader, model, device):
    all_labels = torch.tensor([]).to(device)
    #all_thk = torch.tensor([]).to(device)
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
    #new_directory = os.path.join(data_directory,timestamp + '_trained_model')
    new_directory = os.path.join(data_directory, 'trained_model_'+timestamp)
    os.mkdir(new_directory)
    print("directory created: ", new_directory)

    # save model
    torch.save(network.state_dict(), new_directory + '/trained_model.pth')

    with open(new_directory + '/network.pkl', 'wb') as outp:
        pickle.dump(network, outp, pickle.HIGHEST_PROTOCOL)

    # save NN stats
    # nn_dict = {'network_name':which_network}
    # with open(os.path.join(new_directory,'nn_stats.txt'),'w') as output_file:
    #     output_file.write(str(nn_dict))

    # save image stats
    image_dict = {}
    image_dict.update(input_data.image_stats)
    with open(os.path.join(new_directory, 'image_stats.txt'), 'w') as output_file:
        output_file.write(str(image_dict))

    # save label state
    #label_dict = {'label_norm_code':label_norm_code}
    label_dict = {}
    label_dict.update(input_data.label_stats)
    with open(os.path.join(new_directory, 'label_stats.txt'), 'w') as output_file:
        output_file.write(str(label_dict))

    return
#########################################################


# %%
def main():

    #######################
    # initial inputs
    data_dir = input('paste data directory:  ')
    network_name = input('network name:   ')
    network_to_use = getattr(perf_net, network_name)
    # index_labels = ['d_pix','gap','th_s1','th_s2'] #for setting df index
    use_gpu = False  # manual override for gpu option; having issues with pixel_optim_nn on gpu

    num_epochs = 10  # was 200
    learning_rate = .001  # was 0.001
    momentum = 0.87

    # device = 'cpu'#"cuda" if torch.cuda.is_available() else "cpu"
    if use_gpu == True:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

    print("Using {} device".format(device))

    # turn off/on comutation graph
    torch.set_grad_enabled(True)  # debug 220218 turned off

    input_data = InputData(data_dir)
    input_data.create_datasets(test_perc=0.2,
                               n_batch=2**3,
                               image_norm_code=1,
                               label_norm_code=2)

    #network = perf_net.Network7(input_data,kernel_size = 3).to(device)
    network = network_to_use(input_data, kernel_size=3).to(device)
    network.network_name = network_name

    # print(network)

    # output parameters of model
    params = list(network.parameters())
    print("length of parameters = ", len(params))
    # print("conv1's weight: ",params[0].size())  # conv1's .weight
    print('Trainable parameters:', sum(p.numel()
          for p in network.parameters() if p.requires_grad))

    optimizer_in = optim.Adam(network.parameters(),
                              lr=learning_rate)  # from deeplizard
    # optimizer_in = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum) #from pytorch tutorial
    loss_fn_in = nn.MSELoss()  # MSE seems appropriate for continuous label
    test_error = []
    train_error = []

    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_error = train(input_data.train_dataloader, network,
                            loss_fn_in, optimizer_in, device, train_error)
        test_error = test(input_data.test_dataloader, network,
                          loss_fn_in, device, test_error, error_flag=False)
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

# %%
###########################################

# attempt to use ray
# Uncomment this to enable distributed execution
# `ray.init(address="auto")`


# def train_ray(config):
#     #see: https://docs.ray.io/en/latest/tune/tutorials/tune-tutorial.html
#     model = Network()
#     model.to(device)

#     optimizer_ray = optim.SGD(
#         model.parameters(), lr=config["lr"], momentum=config["momentum"])

#     train_loader = train_dataloader
#     test_loader = test_dataloader

#     for i in range(10):
#         train(train_loader,model, loss_fn, optimizer_ray)
#         acc = test(test_loader, model, loss_fn)

#         print('acc = ', acc)
#         # Send the current training result back to Tune
#         tune.report(score=acc) #had to make sure this was set to mean_loss

#         if i % 5 == 0:
#             # This saves the model to the trial directory
#             torch.save(model.state_dict(), "./model.pth")

# ######################
# search_space = {
#     "lr": tune.loguniform(1e-6,1e-1),#tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
#     "momentum": tune.uniform(0.1, 0.9)
# }

# analysis = tune.run(train_ray, config=search_space)

# print("Best config: ", analysis.get_best_config(
#     metric="score", mode="min"))


##############################################################################
##############################################################################
# normalize data (legacy code)
#stdev = film_df_in.std(axis=0)
#mean = film_df_in.mean(axis=0)
#df_normalized = (film_df_in-film_df_in.mean(axis=0))/film_df_in.std(axis=0)

# norm_values = []
# df_normalized = film_df_in

# for col_name in film_df_in:
#     col_max = df_normalized[col_name].max()
#     col_min = df_normalized[col_name].min()
#     difference = col_max-col_min

#     df_normalized[col_name] = ((df_normalized[col_name]-col_min)/
#                                     (col_max-col_min))

#     norm_values.append([col_name,col_min,col_max])


# film_df_train = df_normalized.sample(frac=0.8)
# film_df_test = df_normalized.drop(film_df_train.index)
