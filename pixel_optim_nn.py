#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:17:17 2022

@author: jaspers2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pickle
import perf_net
import importlib

import matplotlib.pyplot as plt

import image_creation

plt.rcParams['figure.dpi'] = 150


# refs:
# transfer learning: https://debuggercafe.com/transfer-learning-with-pytorch/
# using nn for top opt: chandrasekharTOuNNTopologyOptimization2021


# TODO:
# training original nn on normalized images
# will need to convert predicted image here to non-normalized
# will need statistics from pixel_nn to prep
# also will need function to go backwards
# implement scaling of loss with alpha and material


# %%


class TopNet(nn.Module):
    # Topology neural network. Neural network will take in x,y location
    # Returns two images with shape/normalized thickness as pixel values

    def __init__(self, image_shape, thk_init, vary_thk):

        super(TopNet, self).__init__()
        # super().__init__()

        # (pixel_shape = Num_layers, H_pixels, W_pixels)
        self.image_shape = image_shape
        

        #take in num_pixels_width and num_pixels_height;
        # make an array x of x,y locations based on num_pixels
        # note, this is an array of x,y locations, but my image just has pixel values
        # how to rectify?

        # linear layer output should be 2x125x125

        #self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(20)
        self.bn4 = nn.BatchNorm1d(20)
        # last 3x dims going into reshaping (was 96*496*496 for 500 pixel)
        self.fc1 = nn.Linear(2, 20)
        # last 3x dims going into reshaping (was 96*496*496 for 500 pixel)
        self.fc2 = nn.Linear(20, 20)
        # last 3x dims going into reshaping (was 96*496*496 for 500 pixel)
        self.fc3 = nn.Linear(20, 20)
        # last 3x dims going into reshaping (was 96*496*496 for 500 pixel)
        self.fc4 = nn.Linear(20, 20)
        # last 3x dims going into reshaping (was 96*496*496 for 500 pixel)
        # self.fc5 = nn.Linear(20, 1) #use with sigmoid option
        self.fc5 = nn.Linear(20, 2)  # use with softmax option
        
        self.drop_layer1 = nn.Dropout(0) 
        self.drop_layer2 = nn.Dropout(0)
        self.drop_layer3 = nn.Dropout(0.02) #was 0.02
        self.drop_layer4 = nn.Dropout(0.02) #was 0.02
        
        
        self.l_relu1 = nn.LeakyReLU(0) #was 0.1
        self.l_relu2 = nn.LeakyReLU(0)
        self.l_relu3 = nn.LeakyReLU(0)
        self.l_relu4 = nn.LeakyReLU(0)

        # INITIALIZE WEIGHTS FOR LAYER THICKNESS
        # ref: https://discuss.pytorch.org/t/multiply-parameter-by-a-parameter-in-computation-graph/20401
        # ref: https://stackoverflow.com/questions/67540438/learnable-scalar-weight-in-pytorch

        if vary_thk == True:
            self.weightx = self.thk_layer_init(thk_init)
        elif vary_thk == False:
            self.weightx = self.thk_layer_init(thk_init, grad_setting=False)
            print('WARNING WARNING WARNING: turned requires_grad off on thickness\n')

    def thk_layer_init(self, thk, grad_setting=True):
        # * torch.ones(self.num_layers).reshape((self.num_layers, 1, 1))
        tensor_thk = torch.tensor(thk).reshape((len(thk), 1, 1))
        return torch.nn.Parameter(tensor_thk).requires_grad_(grad_setting)

    def forward(self, x, p_set,symmetric=False):
        # reshaping before linear layers
        #x = x.view(-1, self.num_flat_features(x))
        #x = self.flatten(x)

        #option w/ dropout, but not needed likely with batch norm added
        #prev had drop layer on all fc layers
        x = self.l_relu1(self.drop_layer1(self.fc1(x)))
        x = self.bn1(x)
        x = self.l_relu2(self.drop_layer2(self.fc2(x)))
        x = self.bn2(x)
        x = self.l_relu3(self.drop_layer3(self.fc3(x)))
        x = self.bn3(x)
        x = self.l_relu4(self.drop_layer4(self.fc4(x)))
        x = self.bn4(x)

        #option w/o dropout
        # x = F.relu(self.fc1(x))
        # x = self.bn1(x)
        # x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        # x = F.relu(self.fc3(x))
        # x = self.bn3(x)
        # x = F.relu(self.fc4(x))
        # x = self.bn4(x)

        # legacy ideas, sigmoid option
        #x = torch.sigmoid(self.fc5(x))
        # print(x)

        # softmax option, matches uw
        x = self.fc5(x)
        x = 0.01 + torch.softmax(x, dim=1)
        #print('x after softmax:',x[50:100,:])
        x = x[:, 0].view(-1)
        #print('rho at end:',x.shape)

        #print('rho_mean = ',self.rho.mean())
        #x = self.batch_norm(x)

        if symmetric==True:
            #make symmetric
            C,H,W = self.image_shape
            #get corner, upper triangle
            triu_indices = torch.triu_indices(H//2,W//2)
            corner_image = torch.zeros((H//2,W//2))
            corner_image[triu_indices[0],triu_indices[1]] = x
            
            #fill in corner (xy axis flip)
            corner_image_diag = torch.zeros((H//2,W//2))
            for i in range(H//2):
                corner_image_diag[i,i] = corner_image.diag()[i]
            
            corner_image = corner_image + corner_image.rot90().flip(0) - corner_image_diag

            #extend corner to all 4 quadrants
            x = torch.concat((corner_image,corner_image.flip(1)),axis=1)
            x = torch.concat((x,x.flip(0)),axis=0)

            

        x = torch.reshape(x, (1, self.image_shape[1], self.image_shape[2]))
        self.rho = x  # save density function
        x = x**p_set*self.weightx  # multiply by weights for layer thickness
        return x

    def num_flat_features(self, x):
        # changed to 0 b/c I don't have batches[1:]  # all dimensions except the batch dimension
        size = x.size()[0:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def pretrain():

        return


# %% legacy CustLoss class
# class CustLoss(torch.nn.Module):
#     # ref: https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
#     # ref: https://stackoverflow.com/questions/49821111/pytorch-getting-custom-loss-function-running
#     # ref from rees: https://discuss.pytorch.org/t/custom-loss-functions/29387

#     def __init__(self, centering_factor, denom_factor):
#         super(CustLoss, self).__init__()
        
#         self.centering_factor = centering_factor
#         self.denom_factor = denom_factor

#         # use 0.01 instead of 0 to avoid singularity
#         target_values = np.array([0.01, 1, 1, 0.01, 0.5, 0.5, 340])

#         scaled_target_values = perf_net.norm_labels(target_values,
#                                                     centering_factor,
#                                                     denom_factor)

#         self.target_values = target_values
#         self.scaled_target_values = torch.tensor(scaled_target_values)
#         # create coefficient tensor to use during loss calc.
#         # only care about Tr_ins, Tr_met, Temp

#         # set which labels to include in loss: 1 is yes, 0 is no
#         self.target_coeff = torch.tensor([0, 0, 1, 1, 0, 0, 1])
#         #self.target_coeff = torch.tensor([1, 1, 1, 1, 1, 1, 1])

#     def forward(self, predicted_perf, alpha, rho_mean, vol_target):
#         # predicted performance: tensor [7,]

#         #######
#         # approach 1: full tensor, scaled by target (the "working" approach)
#         error = pow(torch.div(
#             (predicted_perf - self.scaled_target_values), self.scaled_target_values), 2)
#         error_weighted = torch.mul(error, self.target_coeff)

#         loss = torch.sum(error_weighted)
#         vol_constraint = torch.div(rho_mean, vol_target)-1

#         #######
#         # appraoch 2: try error without dividing/normalizing:
#         #error = pow((predicted_perf - self.scaled_target_values), 2)

#         #######
#         # approach 3: break out tr_ins, tr_met, temp components
#         # keep approach 1 active when doing this for now; overwrite loss

#         # scaled predicted perf values

#         # convert scaled predicted perf to unscaled values

#         pred_perf_orig = rescale_tf_labels(predicted_perf)
        
#         #####Create norm_labels function in this code, for pytorch models
        
#         r_ins = pred_perf_orig[0, labels.index('R_ins')]
#         r_met = pred_perf_orig[0, labels.index('R_met')]
#         tr_ins = pred_perf_orig[0, labels.index('Tr_ins')]
#         tr_met = pred_perf_orig[0, labels.index('Tr_met')]
#         a_ins = pred_perf_orig[0, labels.index('A_ins')]
#         a_met = pred_perf_orig[0, labels.index('A_met')]
#         temp = pred_perf_orig[0, labels.index('Temp')]

#         tr_ins_loss = (tr_ins - 1)**2
#         tr_met_loss = (tr_met)**2
#         temp_loss = (temp/340-1)**2

#         # tr_ins_loss = ((tr_ins - 1)/1)**2
#         # tr_met_loss = ((tr_met - 0.01)/0.01)**2
#         # temp_loss = ((min(temp,torch.tensor(340)) - 340)/340)**2

#         # unity check
#         ins_loss = (r_ins + tr_ins + a_ins - 1)**2
#         met_loss = (r_met + tr_met + a_met - 1)**2

#         # loss w/ linear terms (pre-220326)
#         loss = tr_ins_loss + tr_met_loss + temp_loss + ins_loss + met_loss

#         # loss w/ db terms (added 220326)
#         # extinction_ratio,insertion_loss = dB_response(tr_ins, torch.abs(tr_met))
#         #30 dB extinction ratio is sandia target 
#         # loss = insertion_loss**2 + \
#         #     (min(extinction_ratio,torch.tensor(30))-30)**2 + temp_loss + ins_loss + met_loss
#         # print('loss_check:',loss_check)

#         error_terms = [tr_ins_loss.detach().tolist(),
#                        tr_met_loss.detach().tolist(),
#                        temp_loss.detach().tolist(),
#                        ins_loss.detach().tolist(),
#                        met_loss.detach().tolist()]
        
#         error_labels = ['tr_ins',
#                         'tr_met',
#                         'temp',
#                         'ins_loss',
#                         'met_loss']
#         # not working with error_terms portion

#         ########

#         # notes:
#         # alpha multiplies (rho_e*v_e/V*-1). need density array. implement later
#         #loss_tr_ins = (tr_ins/target_tr_ins-1)**2
#         #loss_tr_met = (tr_met/target_tr_met-0)**2
#         #loss_temp = (temp/target_temp-1)**2
#         # loss_vol_constraint = 0#alpha*(self.top_net.rho.mean().item()/self.vol_fract - 1)**2

#         #loss = loss_tr_ins + loss_tr_met + loss_temp + loss_vol_constraint

#         #print('loss:', loss.tolist())

#         # error_terms = []

#         # for value in error_weighted.tolist()[0]:
#         #     if value > 0:
#         #         error_terms.append(value)

#         # print('Tr_ins = ',tr_ins.tolist(), ', %=',loss_tr_ins.tolist(),
#         #       '\nTr_met = ',tr_met.tolist(),', %=',loss_tr_met.tolist(),
#         #       '\nTemp = ',temp.tolist(),', %=',loss_temp.tolist())
#         return loss, vol_constraint, error_terms, error_labels

# %%
class Labels():
    
    
    def __init__(self,perfnn):
        
        self.label_center = torch.tensor(perfnn.label_centering_factor)
        self.label_denom = torch.tensor(perfnn.label_denom_factor)
        
    def label_update(self,label_tf, state):
        """assign updated labels

        Parameters
        ----------
        label_tf : TORCH.TENSOR
            TENSOR OF LABELS
        state : STR
            "normalized" OR "real"

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if type(label_tf) == False:
            raise Exception('label must be tensor')

        
        if state == 'real' or state == 'normalized':
            pass
        else:
            raise Exception('state must be normalized or real')
        
            
        self.labels = label_tf
        self.state = state
        
    def normalize_labels(self):
        if self.state == 'real':
            self.labels = (self.labels-self.label_center)/self.label_denom
            self.state = 'normalized'
        
    def scale_labels(self):
        if self.state == 'normalized':
            self.labels = (self.labels*self.label_denom + self.label_center)
            self.state = 'real'
        
    #add function to convert back and forth
    #add flag: scaled or normalized
    

#%%
class TopOpt():
    """
    Topology optimization. Executes topology optimization using 
    pre-trained performance network and TopOpt topology neural network.
    """
    def __init__(self, perfnn, learning_rate, device, vary_thk, symmetric=False):
        self.perfnn = perfnn
        self.learning_rate = learning_rate
        self.device = device
        self.vary_thk = vary_thk
        self.symmetric = symmetric
        
        
        torch.set_grad_enabled(True)
        
        N,C,H,W = perfnn.image_stats['image_shape']
        self.image_shape = (C,H,W)
        
        # create initial input image for TopNet
        if symmetric == False:
            self.input_xy = self.generate_xy().to(device)
        elif symmetric == True:
            self.input_xy = self.gen_xy_sym().to(device)
        
        print(self.input_xy.shape)
        
        #lock down perfnn
        self.perfnn.requires_grad_(False) #need to verify this works
        
        #initialize topnet
        self.top_net = TopNet(self.image_shape, (1.,1.), vary_thk)#(0.5, 0.49))
        

        
        self.optimizer = optim.Adam(
            self.top_net.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def set_targets(self, labels, target_tr_ins, target_tr_met, target_temp):
        """set targets for labels, Tr and Temp
        """
        
        target_labels = Labels(self.perfnn)
        
        array = np.zeros(len(labels))
        array[labels.index('Tr_ins')] = target_tr_ins
        array[labels.index('Tr_met')] = target_tr_met
        array[labels.index('Temp')] = target_temp
        array_tf = torch.tensor(array)
        
        target_labels.label_update(array_tf, 'real')
        
        self.target_labels = target_labels
        

    def generate_xy(self):
        """Generate xy input points for top_net

        Parameters
        ----------
        none

        Returns
        -------
        xy_tensor : tensor [:,2]
            tensor of x,y locations

        """
        C,H,W = self.image_shape
        
        x_loc = [i/W for i in range(W)]
        y_loc = [i/H for i in range(H)]

        # need grid combo, not just all points
        combined_xy = [[x, y] for x in x_loc for y in y_loc]

        return torch.tensor(combined_xy)#, requires_grad=True)
    
    def gen_xy_sym(self):
        #generates indices for upper triangle
        C,H,W = self.image_shape
        xy = np.triu_indices(H/2)
        combined_xy = [i for i in zip((xy[0]/H/2).tolist(),(xy[1]/H/2).tolist())]
        
        return torch.tensor(combined_xy)


    def cust_loss(self, pred_perf_in, alpha):
        #############################
        #legacy code, without db calc
        
        # target = self.target_labels.labels
        # pred = pred_perf.labels.reshape(-1)
        # labels = self.perfnn.label_names
        
        # target_Tr_ins = target[labels.index('Tr_ins')]
        # target_Tr_met = target[labels.index('Tr_met')]
        # target_Temp = target[labels.index('Temp')]
        
        # pred_Tr_ins = pred[labels.index('Tr_ins')]
        # pred_Tr_met = pred[labels.index('Tr_met')]
        # pred_Temp = pred[labels.index('Temp')]
        
        
        
        # loss = ((pred_Tr_ins - target_Tr_ins)**2 
        #                 + (pred_Tr_met - target_Tr_met)**2 
        #                 )#+ (pred_Temp - target_Temp)**2)
        
        
        # error_terms = [abs(target_Tr_ins - pred_Tr_ins).detach().tolist(),
        #                abs(target_Tr_met - pred_Tr_met).detach().tolist(),
        #                abs(target_Temp - pred_Temp).detach().tolist()]
        
        # error_terms_labels = ['Tr_ins','Tr_met','Temp']
        
        # # print("targets: Tr_ins, Tr_met, Temp: ", target_Tr_ins, target_Tr_met, target_Temp)
        # # print("predicted: Tr_ins, Tr_met, Temp: ", pred_Tr_ins, pred_Tr_met, pred_Temp)
        # # print("loss:",loss)
        # return loss, error_terms, error_terms_labels
        #############################
        target = self.target_labels.labels
        pred_perf_in.scale_labels()
        pred = pred_perf_in.labels.reshape(-1)
        labels = self.perfnn.label_names
        
        target_Tr_ins = target[labels.index('Tr_ins')]
        target_Tr_met = target[labels.index('Tr_met')]
        target_Temp = target[labels.index('Temp')]
        
        pred_Tr_ins = pred[labels.index('Tr_ins')]
        pred_Tr_met = pred[labels.index('Tr_met')]
        pred_Temp = pred[labels.index('Temp')]
        
        pred_ext_ratio, pred_ins_loss = dB_response(pred_Tr_ins, pred_Tr_met)
        
        loss = (30-pred_ext_ratio) + (pred_ins_loss-3)
        
        
        error_terms = [abs(target_Tr_ins - pred_Tr_ins).detach().tolist(),
                        abs(target_Tr_met - pred_Tr_met).detach().tolist(),
                        abs(target_Temp - pred_Temp).detach().tolist()]
    
        error_terms_labels = ['Tr_ins','Tr_met','Temp']
        
        return loss, error_terms, error_terms_labels


    def optimize(self, alpha_0, delta_alpha, alpha_max, max_epochs, p_init, delta_p, p_max):
        """perform optimization/training of top_op

        Positional arguments:
            max_epochs : int 
                maximum number of epochs (int)

        """

        self.perfnn.eval()  # only set perf_net to eval mode
        self.top_net.train()

        alpha = alpha_0

        optimize_loss = []
        error_terms = []
        
        self.target_labels.normalize_labels()
        predicted_perf = Labels(self.perfnn)

        p_set = p_init
        
        for i in range(max_epochs):
                
            # forward pass to obtain predicted images
            # no need to normalize images first b/c using "normalized" layer thicknesses
            
            images = self.top_net(self.input_xy, p_set, symmetric=self.symmetric)  # tensor [N_batch,2]
            images = images[None]  # adds axis, [1,2,20,20]
            predicted_perf.label_update(self.perfnn(images), 'normalized')
            
            #this is my current "best guess"
            objective, error_terms_in, error_labels_in = self.cust_loss(predicted_perf, alpha)

            #backpropogation
            self.optimizer.zero_grad()
            loss = objective  # + alpha*pow(vol_constraint,2)
            loss.backward()
            self.optimizer.step()

            error_terms.append(error_terms_in)

            # increment counters
            alpha += delta_alpha
            alpha = min(alpha, alpha_max)
            p_set = min(p_set+delta_p, p_max)

            ##########
            # print thicknesses (rescale first!)
            thicknesses = np.array(self.top_net.weightx.tolist())
            thicknesses_scaled = self.rescale_thk(thicknesses)

            if i % 100 == 0 or i == max_epochs-1:
                print('---------------------')
                print('epoch: ', i)
                print('objective loss: ', objective.tolist())
                print('loss = ', loss.tolist())
                print('Thicknesses = ', thicknesses_scaled)
            ##########

            optimize_loss.append(loss.tolist())

            if i == 1:
                self.plot_images(images, 'optimized design after 1 epoch')

        # get/show final perforamcne
        self.top_net.eval()
        images = self.top_net(self.input_xy, p_set, symmetric=self.symmetric)  # tensor [N_batch,2]
        images = images[None]  # adds axis, [1,2,20,20]
        predicted_perf.label_update(self.perfnn(images), 'normalized')        


        
        predicted_perf.scale_labels()
        self.predicted_perf = predicted_perf
        self.images = images
        self.loss = optimize_loss
        self.plot_images(images, 'final image')
        self.error_terms = error_terms
        self.error_labels = error_labels_in

       
    def print_predicted_performance(self): 
        #print('Final design pred perf:\n',labels,'\n',self.predicted_perf)
        print('--------')
        print('Final design pred perf:')
        
        labels = self.perfnn.label_names
        pred_perf = self.predicted_perf.labels.detach().tolist()[0]
        
        for i in range(len(labels)):
            print(labels[i], ':', pred_perf[i])
        print('--------')
        print('unity check:')
        print('ins: R_ins + Tr_ins + A_ins = ',
              pred_perf[labels.index('R_ins')] +
              pred_perf[labels.index('Tr_ins')] +
              pred_perf[labels.index('A_ins')])

        print('met: R_met + Tr_met + A_met = ',
              pred_perf[labels.index('R_met')] +
              pred_perf[labels.index('Tr_met')] +
              pred_perf[labels.index('A_met')])
        
        ext_ratio, ins_loss = dB_response(pred_perf[labels.index('Tr_ins')],
                                          abs(pred_perf[labels.index('Tr_met')]))
        print('Extinction ratio (goal: > 30 dB): ',ext_ratio)
        print('Insertion loss (goal: < 3 dB): ',ins_loss)
    

    def rescale_thk(self, np_array):
        # takes in np array (N_layers), rescales and returns np array
        np_array = np_array.reshape(-1)
        thk = np.array(self.perfnn.image_stats['max'])
        #denom = self.perfnn.image_denom_factor.reshape(-1)
        #center = self.perfnn.image_centering_factor.reshape(-1)
        thk_rescaled = np_array*thk

        return thk_rescaled

    def plot_images(self, images, title_str):
        #print('images shape: ', np.shape(images.detach().numpy()))

        # only plot one channel, to get view of distribution
        # second channel will be same but different thickness
        # for some reason, only plotting at end. need to debug

        np_images = images.detach().numpy()
        plt.imshow(np_images[0][0])
        plt.xlabel('pixel')
        plt.ylabel('pixel')
        plt.title(title_str)
        plt.show()

        return

    def save_results(self, path):
        #get timestamp
        timestamp = image_creation.create_timestamp()
        
        # save image
        image = image_creation.Image()
        image.images = self.images.detach().numpy()[0][0]
        H,W = image.images.shape
        name = path + '/optimized_design_' + timestamp + '.npy'
        np.save(name, image.images)
        print('saved to: ', name)
        
        #save bit files
        image.images[image.images>0.5] = 1
        image.images[image.images<1] = 0
        
        #save requires N,C,H,W images
        image.images = image.images.reshape(1,1,H,W)
        
        image.save_comsol_inputs_gen2(path, timestamp)


# %%
def use_gpu(use_gpu):
    if use_gpu == True:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

    print("Using {} device".format(device))

    return device


def load_perfnet(folder):
    with open(folder + '/network.pkl','rb') as inp:
        nn = pickle.load(inp)
        return nn

#%% legacy import stats
# def import_stats(perf_nn_folder):
#     """ import stats and label code from label_stats.txt """

#     with open(perf_nn_folder + '/image_stats.txt') as f:
#         image_stats_in = eval(f.read())

#     with open(perf_nn_folder + '/label_stats.txt') as f:
#         label_stats_in = eval(f.read())

#     return image_stats_in, label_stats_in
#%%

def dB_response(Tr_ins,Tr_met):   
    '''Calculates extinction ratio and insertion loss
    

    Parameters
    ----------
    Tr_ins : TORCH.TENSOR, NP.ARRAY OR INT
        INSULATING TRANSMITTANCE
    Tr_met : TORCH.TENSOR, NP.ARRAY OR INT
        METALLIC TRANSMITTANCE

    Returns
    -------
    extinction_ratio
    insertion_loss

    '''
        
    if torch.is_tensor(Tr_ins) or torch.is_tensor(Tr_met) == True:
        extinction_ratio = 10*torch.log10(Tr_ins/Tr_met)
        insertion_loss = 10*torch.log10(1/Tr_ins)
    else:
        extinction_ratio = 10*np.log10(Tr_ins/Tr_met)
        insertion_loss = 10*np.log10(1/Tr_ins)
    
    return(extinction_ratio,insertion_loss)

def get_folder():
    # folder = '/home/jaspers2/Documents/pixel_optimization/validation_data/220328/2022-3-28-12-56_2/2022-3-31-11-18_trained_model'
    # folder_input = input('use existing folder? y or n \n' + folder + '\n')
    # if folder_input == 'n':
    #     folder = input('list folder: ')
    #     print('using folder: \n',folder)
    
    folder = input('base model folder: ')
    return folder

def plot_error(error_terms, error_labels, y_limit = None):
    error_terms = np.array(error_terms)
    for i in range(len(error_labels)):
        plt.plot(error_terms[:, i], label=error_labels[i])
    
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss')
    plt.title('Breakdown of loss contributions')
    plt.legend()
    if y_limit != None:
        plt.ylim(0, y_limit)
    plt.show()
    
def compare_prediction(pred_image, base_model_folder):
    #target = np.load(input('target file: '))
    target_file = base_model_folder + '/target_image.npy'
    target = np.load(target_file)
    
    pred_image = pred_image.detach().numpy()
    
    pred_image[pred_image>=0.5] = 1
    pred_image[pred_image<0.5] = 0
    
    num_pixels = len(pred_image.reshape(-1))
    
    error = abs(pred_image - target)
    error_total = int(error.reshape(-1).sum())
    perc_error = error_total/num_pixels
    
    
    plt.imshow(target)
    plt.title('target image')
    
    
    print('number incorrect = ',error_total)
    print('percent_error = ',perc_error)
    

#%% Legacy main code
# def legacy_code_main():

#     image_stats, label_stats = import_stats(perf_nn_folder)
    
#     # pytorch follows convention: N_batch, N_chnls, N_height, N_width
    
    
#     #image stats is missing stdev, not sure if needed

    
#     seed_hyperparams = {'learning_rate': 1, 
#                         'max_epochs':3000,
#                         'p_init':1,  #p=0 makes it solid film, p=1 is no penalty
#                         'p_max': 1,#4, 
#                         'delta_p': 0.005}
    
#     hyperparams = {'learning_rate': 0.01, 
#                    'vol_fraction': 0.5,
#                    'vary_thk': False,
#                    'p_init':1,  #p=0 makes it solid film, p=1 is no penalty
#                    'p_max': 4, 
#                    'delta_p': 0.005}
    
#     topopt_params = {'alpha_max': 0,  # 1,
#                      'delta_alpha': 0,  # 0.05,
#                      'alpha_0': 0,  # 0.1,
#                      'eps_g_star': 0.035,
#                      'max_epochs': 2000,
#                      }

    
#     # train optim nn
#     top_opt = TopOpt(image_params)
#     top_opt.initialize_optimizer(hyperparams)
#     top_opt.optimize()
#     print('optimize done')
    
#     print('WARNING WARNING WARNING: Selected perf_net is hardcoded')
    
#%%
def main():
    # set True to use GPU, False to use CPU
    print("Warning: changed loss function")
    device = use_gpu(False)
    
    base_model_folder = get_folder()
    #base_model_folder = '/home/jaspers2/Documents/pixel_optimization/dof_exploration/testing/220506-0710'
    perf_nn_folder = base_model_folder + '/trained_model'
    
    
    # only use during debugging (slows code)
    torch.autograd.set_detect_anomaly(True)
    
    
    #load perf_nn
    perfnn = load_perfnet(perf_nn_folder)
        
    #initilize top_opt
    top_opt = TopOpt(perfnn, .00001, device, False, symmetric=True)
    top_opt.set_targets(perfnn.label_names, 1,0,340)
    top_opt.optimize(0,0,0,10_000,1,0.005,1)
    top_opt.print_predicted_performance()
        
    plt.plot(np.array(top_opt.loss))
    plt.title('Optimizing loss')
    plt.show()
    
    plt.hist(top_opt.top_net.rho.detach().numpy().reshape(-1))
    plt.title(r'Distribution of $\rho$ ')
    plt.show()
    
    plot_error(top_opt.error_terms, top_opt.error_labels)
    
    
    # if input('compare with target? y to compare:  ') == 'y':
    #     compare_prediction(top_opt.images, base_model_folder)
    
    if input('save results? y to save:  ') == 'y':
        top_opt.save_results(base_model_folder)

    return top_opt


if __name__ == '__main__':
    top_opt = main()
    
    