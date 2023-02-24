#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import perf_net
import matplotlib.pyplot as plt
import image_creation
import os

plt.rcParams['figure.dpi'] = 150

# ref: chandrasekharTOuNNTopologyOptimization2021

class TopNet(nn.Module):
    """
    Topology neural network. Neural network will take in x,y location
    Returns two images with shape/normalized thickness as pixel values
    (pixel_shape = Num_layers, H_pixels, W_pixels)
    """
    
    def __init__(self, image_shape, thk_init, vary_thk):
        """
        """
        super(TopNet, self).__init__() 
        self.image_shape = image_shape
        
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(20)
        self.bn4 = nn.BatchNorm1d(20)
        self.bn5 = nn.BatchNorm1d(20)
        self.fc1 = nn.Linear(2, 20)

        nn.init.xavier_normal_(self.fc1.weight) #xavier_normal_from TOuNN.py
        self.fc2 = nn.Linear(20, 20)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(20, 20)
        nn.init.xavier_normal_(self.fc3.weight)
        self.fc4 = nn.Linear(20, 20)
        nn.init.xavier_normal_(self.fc4.weight)
        self.fc5 = nn.Linear(20, 2)  # use with softmax option
        nn.init.xavier_normal_(self.fc5.weight)
                
        self.l_relu1 = nn.LeakyReLU(0.01) #was 0.1
        self.l_relu2 = nn.LeakyReLU(0.01)
        self.l_relu3 = nn.LeakyReLU(0.01)
        self.l_relu4 = nn.LeakyReLU(0.01)
        self.l_relu5 = nn.LeakyReLU(0.01)


        # INITIALIZE WEIGHTS FOR LAYER THICKNESS
        # ref: https://discuss.pytorch.org/t/multiply-parameter-by-a-parameter-in-computation-graph/20401
        # ref: https://stackoverflow.com/questions/67540438/learnable-scalar-weight-in-pytorch

        if vary_thk == True:
            self.weightx = self.thk_layer_init(thk_init)
        elif vary_thk == False:
            self.weightx = self.thk_layer_init(thk_init, grad_setting=False)
            print('WARNING WARNING WARNING: thickness fixed; turned requires_grad off on thickness\n')

    def thk_layer_init(self, thk, grad_setting=True):
        """takes normalized initial thk array, creates weightx NN parameters
        """
        
        tensor_thk = torch.tensor(thk).reshape((len(thk), 1, 1))
        return torch.nn.Parameter(tensor_thk).requires_grad_(grad_setting)

    def forward(self, x, p_set, symmetric=False):        
        #option w/o dropout
        x = self.bn0(x)
        x = self.l_relu1(self.fc1(x))
        x = self.bn1(x)
        x = self.l_relu2(self.fc2(x))
        x = self.bn2(x)
        x = self.l_relu3(self.fc3(x))
        x = self.bn3(x)
        x = self.l_relu4(self.fc4(x))
        x = self.bn4(x)
        x = self.fc5(x)

        # softmax option
        x = 0.001 + torch.softmax(x, dim=1)
        x = x[:, 0].view(-1) #keep order but remove extra axis
        x = torch.reshape(x, (1, self.image_shape[1], self.image_shape[2]))
        self.rho = x  # save density function
        
        # multiply by weights for layer thickness (creates indiv channels)
        x = torch.pow(x,p_set)*self.weightx
        return x


class Labels():    
    def __init__(self,perfnn):
        """
        """
        self.label_stats = perfnn.label_stats
        
    def label_update(self,label_tf, state):
        """
        update label state (normalized or real)
        
        :param label_tf: tensor of labels
        :type label_tf: torch.tensor
        :param state: state of label; "normalized" OR "real"
        :type state: str
        :raises Exception: label must be a tensor
        """

        if type(label_tf) != torch.Tensor:
            raise Exception('label must be tensor')

        
        if state == 'real' or state == 'normalized':
            pass
        else:
            raise Exception('state must be normalized or real')
        
            
        self.labels = label_tf.reshape((1,-1)) #reshape for perfnet 
        self.state = state
        
    def normalize_labels(self):
        if self.state == 'real':
            self.labels = perf_net.norm_labels(self.labels,self.label_stats)
            self.state = 'normalized'
        
    def scale_labels(self):
        if self.state == 'normalized':
            self.labels = perf_net.rescale_labels(self.labels,self.label_stats)
            self.state = 'real'

class TopOpt():
    """
    Topology optimization. Executes topology optimization using 
    pre-trained performance network and TopOpt topology neural network.
    """
    def __init__(self, perfnn, learning_rate, device, vary_thk, symmetric=False):
        """initialize TopOpt
        """
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
        
        #lock down perfnn
        self.perfnn.requires_grad = False
        for param in self.perfnn.parameters():
            param.requires_grad = False
        
        #initialize topnet
        self.top_net = TopNet(self.image_shape, (1.,1.), vary_thk)#(0.5, 0.49))
              
        self.optimizer = optim.Adam(
            self.top_net.parameters(), amsgrad=True, lr=self.learning_rate)#, weight_decay=1e-5)
        
    def set_targets(self, labels, targets):
        """set targets for labels: ext_ratio, Temp or dT
        """
        
        target_labels = Labels(self.perfnn)
        array = np.zeros(len(labels))
        
        for i,name in enumerate(labels):
            array[i] = targets[i]    
        
        array_tf = torch.tensor(array)     
        target_labels.label_update(array_tf, 'real')
        
        self.target_labels = target_labels
        

    def generate_xy(self):
        """
        Generate xy input points for top_net
        
        :return: xy_tensor (N,2) tensor of x,y locations
        :rtype: tensor

        """
        C,H,W = self.image_shape
        
        x_loc = [i/W for i in range(1,W+1)]
        y_loc = [i/H for i in range(1,H+1)]

        # need grid combo, not just all points
        combined_xy = [[y, x] for y in y_loc for x in x_loc]
        return torch.tensor(combined_xy)

    def cust_loss(self, pred_perf_in, alpha):
        #############################
        #legacy code, without db calc
        
        target = self.target_labels.labels
        pred = pred_perf_in.labels
        labels = self.perfnn.label_names
        
        target_ext_ratio = target[0,labels.index('ext_ratio')]
        target_Temp = target[0,labels.index('Temp')]
        
        pred_ext_ratio = pred[0,labels.index('ext_ratio')]
        pred_Temp = pred[0,labels.index('Temp')]
        
        loss = (torch.square((target_ext_ratio-pred_ext_ratio)/target_ext_ratio) + 
                torch.square((target_Temp-pred_Temp)/target_Temp))
        
        
        error_terms = [abs(target_ext_ratio - pred_ext_ratio).detach().tolist(),
                        abs(target_Temp - pred_Temp).detach().tolist()]
        
        error_terms_labels = ['ext_ratio','Temp']
                
        return loss, error_terms, error_terms_labels
    
    def cust_loss_dT(self, pred_perf_in, alpha):
        target = self.target_labels.labels
        pred = pred_perf_in.labels
        labels = self.perfnn.label_names
        
        target_ext_ratio = target[0,labels.index('ext_ratio')]
        target_Temp = target[0,labels.index('dT')]
        
        pred_ext_ratio = pred[0,labels.index('ext_ratio')]
        pred_Temp = pred[0,labels.index('dT')]
       
        loss = (torch.square((target_ext_ratio-pred_ext_ratio)/target_ext_ratio) + 
                torch.square((target_Temp-pred_Temp)/target_Temp))

        error_terms = [abs(target_ext_ratio - pred_ext_ratio).detach().tolist(),
                        abs(target_Temp - pred_Temp).detach().tolist()]
        
        error_terms_labels = ['ext_ratio','dT']
        
        return loss, error_terms, error_terms_labels

    def pretrain(self, initial_density, num_epochs):
        """pretrain top_opt to output given, uniform density
        """
        self.perfnn.eval()  #set perf_net to eval mode
        self.top_net.train() #top_net is training (influences dropout)
        
        #make target rho array
        C,H,W = self.image_shape
        target = self.top_net.weightx*float(initial_density)*torch.ones(1,int(H),int(W)) #density funct is 1 channel only
        target = target[None]
        
        for i in range(num_epochs):
            images = self.top_net(self.input_xy, 1,symmetric=True)
            images = images[None]
            loss = ((target-images)**2).sum()
            
            if i % 100 == 0:
                print("pretrain loss: ",loss)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.plot_images(images, 'pretrained image')

    
    def optimize(self, alpha_0, delta_alpha, alpha_max, max_epochs, p_init, delta_p, p_max):
        """perform optimization/training of top_op

        """

        self.perfnn.eval()  #set perf_net to eval mode
        self.top_net.train() #top_net is training (influences dropout)
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
            images = images[None]  #add axis
            if i == 0:
                self.plot_images(images, 'initial image')
            pred_label = self.perfnn(images)
            predicted_perf.label_update(pred_label, 'normalized')
            
            #this is current "best guess"
            if 'Temp' in self.perfnn.label_names:
                objective, error_terms_in, error_labels_in = self.cust_loss(predicted_perf, alpha)
            elif 'dT' in self.perfnn.label_names:
                objective, error_terms_in, error_labels_in = self.cust_loss_dT(predicted_perf, alpha)

            #backpropogation
            self.optimizer.zero_grad()
            loss = objective
            loss.backward(retain_graph=True)
            self.optimizer.step()

            error_terms.append(error_terms_in)

            # increment counters
            alpha += delta_alpha
            alpha = min(alpha, alpha_max)
            p_set = min(p_set+delta_p, p_max)

            ##########
            # print rescaled thicknesses
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
        images = images[None]  # adds axis
        predicted_perf.label_update(self.perfnn(images), 'normalized')        

        predicted_perf.scale_labels()
        self.predicted_perf = predicted_perf
        self.images = images
        self.loss = optimize_loss
        self.plot_images(images, 'final image')
        self.error_terms = error_terms
        self.error_labels = error_labels_in

       
    def print_predicted_performance(self): 
        print('--------')
        print('Final design pred perf:')
        
        labels = self.perfnn.label_names
        pred_perf = self.predicted_perf.labels.detach().tolist()[0]
        
        for i in range(len(labels)):
            print(labels[i], ':', pred_perf[i])
        print('--------')    

    def rescale_thk(self, np_array):
        # takes in np array (N_layers), rescales and returns np array
        np_array = np_array.reshape(-1)
        thk = np.array(self.perfnn.image_stats['max'])
        thk_rescaled = np_array*thk

        return thk_rescaled

    def plot_images(self, images, title_str):
        # only plot one channel, to get view of distribution
        # second channel will be same but different thickness
        
        np_images = images.detach().numpy()
        plt.imshow(np_images[0][0])
        plt.xlabel('pixel')
        plt.ylabel('pixel')
        plt.title(title_str)
        plt.show()

        return

    def save_results(self, path):
        timestamp = image_creation.create_timestamp()
        
        #make folder
        new_directory = os.path.join(path, 'optimized_design_'+timestamp)
        os.mkdir(new_directory)
        print("directory created: ", new_directory)
        
        #save top_opt
        torch.save(self.top_net.state_dict(), new_directory + '/trained_model.pth')
        with open(new_directory + '/top_opt_network.pkl', 'wb') as outp:
            pickle.dump(self.top_net, outp, pickle.HIGHEST_PROTOCOL)
        
        # save image
        image = image_creation.Image()
        image.images = self.images.detach().numpy()[0][0]
        
        #make symmetric (base image is upper left corner)
        image.images = np.concatenate((image.images,np.flip(image.images,0)),axis=0)
        image.images = np.concatenate((image.images,np.flip(image.images,1)),axis=1)
        
        H,W = image.images.shape
        name = new_directory + '/optimized_design_' + timestamp + '.npy'
        np.save(name, image.images)
        print('saved to: ', name)
        
        #convert to binary for csv and bit files
        image.images[image.images>0.5] = 1
        image.images[image.images<1] = 0
        
        #save csv for fab
        image_txt = []
        for i in range(H):
            for j in range(W):
                image_txt.append([i,j,image.images[i,j]])
        np.savetxt(new_directory+'/optimized_design_'+timestamp+'.csv',image_txt,delimiter=",")
        
        #save bits (requires N,C,H,W images)
        image.images = image.images.reshape(1,1,H,W)
        image.save_comsol_inputs(new_directory, timestamp)
        
        #save readme with settings
        readme = ""
        self.target_labels.scale_labels()
        targets = self.target_labels.labels
        readme += f'target values: {targets} \n'
        
        with open(os.path.join(new_directory,'readme.txt'),'w') as output:
            output.write(readme)


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


def dB_response(Tr_ins,Tr_met): 
    """
    Calculates extinction ratio and insertion loss
    
    :param Tr_ins: insulating transmittance
    :type Tr_ins: torch.tensor, np.array or int
    :param Tr_met: metallic transmittance
    :type Tr_met: torch.tensor, np.array or int
    :return: extinction ratio, insertion loss
    :rtype: int, int

    """
        
    if torch.is_tensor(Tr_ins) or torch.is_tensor(Tr_met) == True:
        extinction_ratio = 10*torch.log10(Tr_ins/Tr_met)
        insertion_loss = 10*torch.log10(1/Tr_ins)
    else:
        extinction_ratio = 10*np.log10(Tr_ins/Tr_met)
        insertion_loss = 10*np.log10(1/Tr_ins)
    
    return(extinction_ratio,insertion_loss)

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
    
    
def vis_model_weights(model,layer):
    """plot model weights
    
    ref:
    https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
    """
    
    model_weights = []
    conv_layers = []
    
    model_children = list(model.conv_layers.children())
    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) ==  nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
    
    
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[layer]):
        plt.subplot(6, 5, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.show()


def top_opt_funct(perf_nn_folder, 
                  target_choice = 1, 
                  target_db = 10, 
                  print_details=False):
    # set True to use GPU, False to use CPU
    device = use_gpu(False)  
        
    # only use during debugging (slows code)
    torch.autograd.set_detect_anomaly(True)
    
    #load perf_nn
    perfnn = load_perfnet(perf_nn_folder)
    
    #initilize top_opt
    top_opt = TopOpt(perfnn, .001, device, False, symmetric=False)
    
    if print_details == True:
        print('Trainable parameters:', sum(p.numel()
            for p in top_opt.top_net.parameters() if p.requires_grad))
        
        
    if target_choice == '1':
        #for use with actual data
        if 'Temp' in perfnn.label_names:
            #orig Temp selection was 10 dB, 285 K
            top_opt.set_targets(perfnn.label_names, (target_db,285))
        elif 'dT' in perfnn.label_names:
            #modified to use dT instead of temp
            top_opt.set_targets(perfnn.label_names, (target_db, 15))
    elif target_choice == '2':
        #for dummy data, slightly different targets
        top_opt.set_targets(perfnn.label_names, (20, 10))    
    
    num_epochs = 3_000
    p_max = 2
    top_opt.optimize(0,0,0,num_epochs,1,(p_max-1)/num_epochs,p_max)
    top_opt.print_predicted_performance()

    if print_details == True:    
        plt.plot(np.array(top_opt.loss))
        plt.title('Optimizing loss')
        plt.show()
        
        plt.hist(top_opt.top_net.rho.detach().numpy().reshape(-1))
        plt.title(r'Distribution of $\rho$ ')
        plt.show()
    
        plot_error(top_opt.error_terms, top_opt.error_labels)
    
    if target_choice == '2': #if dummy data selection
        if input('compare with target (dummy data only)? y to compare:  ') == 'y':
            base_model_folder = input('base model folder: ')
            compare_prediction(top_opt.images, base_model_folder)

    return top_opt


def main():
    perf_nn_folder = input('trained_model_[date] folder: ')
    target_choice_in = input('1) Actual or 2) dummy data?  ')
    top_opt = top_opt_funct(perf_nn_folder, 
                  target_choice = target_choice_in,
                  print_details = True)
    if input('save results? y to save:  ') == 'y':
        top_opt.save_results(perf_nn_folder)


if __name__ == '__main__':
    top_opt_out = main()