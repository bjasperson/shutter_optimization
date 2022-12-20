#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import datetime
import os
import itertools
import matplotlib.pyplot as plt


class Image():
    
    def __init__(self):
        self.thk = [400,240] #thk[0] is closest to substrate
        self.temp = [273.15, 373.15]
        
    def input_selection(self):
        input_list = input('which input list?\n' + 
                           '1=full permutation, generate data\n'+
                           '2=random images, generate data\n'+
                           '3=2 um wide, 0.1 um pixels, 20x20 random symmetric images, generate data\n'+
                           '4=checkerboard\n'
                           '******\n')
        
        if input_list == '1':
            input_width_pixels = int(input('number of pixels wide (4 or less!): '))
            if input_width_pixels > 4:
                raise Exception('Must be 4 or less')
            self.all_permutations(input_width_pixels)
            
        elif input_list == '2':
            input_width_pixels = int(input('number of pixels wide: '))
            input_N_points = int(input('number of images to gen: '))
            self.images = random_gen_2(input_N_points,input_width_pixels)
                      
        elif input_list == '3':
            input_width_pixels = 20
            self.cell_width_um = 2
            self.pixel_width_um = 0.1
            input_N_points = int(input('number of images to gen: '))
            self.images = random_gen_4(input_N_points,input_width_pixels)
        
        elif input_list == '4':
            input_width_pixels = 20
            self.cell_width_um = 2
            self.pixel_width_um = 0.1
            self.images = checkboard(input_width_pixels)
        else:
            raise Exception('Select available input list')
            
    
    def all_permutations(self,N_pixels_width):
        #returns a 2**num_pixels x num_thk x num_pixels_width x num_pixels_width
        N_pixels = N_pixels_width**2
        N_pts = 2**N_pixels
        
        #https://stackoverflow.com/questions/14931769/how-to-get-all-combination-of-n-binary-value
        image_list = list(itertools.product([0,1],repeat = N_pixels))
        image_array = np.array(image_list).reshape(N_pts,1,N_pixels_width,N_pixels_width)
        
        self.images = image_array
        
    def create_thk_images(self):
        #convert binary images to pixel "thickness" images
        thk_array = np.array(self.thk).reshape(len(self.thk),1,1)
        self.images_w_thk = self.images*thk_array
    
    def save_output(self):
        feature_image = self.images_w_thk
        thk = self.thk
        timestamp = create_timestamp()
        master_directory = input('directory for timestamped folder creation:')
        path = folder_create(timestamp,master_directory)
        np.save(os.path.join(path,timestamp + '_images'),feature_image)
        
        #readme
        comsol_rev = input('comsol model revision: ')
        readme_file = readme_create(comsol_rev,feature_image.shape,thk)
        with open(os.path.join(path,'readme.txt'),'w') as output:
            output.write(readme_file)
        
        #param file for comsol
        self.save_comsol_inputs(path, timestamp)
        
        #simulated results
        if feature_image.shape[2] == 20:
            if input('generate (20x20 symmetric) simulated results? y to gen/save:   ') == 'y':
                target_image, labels, fake_results = simulated_sym_results(self.images)
                target_image = target_image[:10,:10]
                
                reduced_images = feature_image[:,:,:10,:10]
                np.save(os.path.join(path,timestamp + '_reduced_images'),reduced_images)
                
                np.savetxt(os.path.join(path,'final_comsol_results.csv'),fake_results,header=labels,comments = '',delimiter=",")
                np.save(os.path.join(path,'target_image'),target_image)
        

    def save_comsol_inputs(self, path, timestamp):
        N,C,H,W = self.images.shape
        
        label = ['bit'+"_"+str(i)+"_"+str(j) for i in range(H) for j in range(W)]
        bits = self.images.reshape(N,-1)
        
        date,time = timestamp.split("-")
        
            
        #bits file
        os.mkdir(os.path.join(path,'bits'))
        for j in range(N):            
            bits_file = ""
            for i,name in enumerate(label):
                bits_file += name + " " + str(bits.transpose()[i,j].tolist()) + "\n"    
            
            with open(os.path.join(path,'bits/'+timestamp+'-'+str(j)+'_bits.txt'),'w') as f:
                f.write(bits_file)
       

def random_gen_2(N_pts,N_pixels_width):
    rand_array = np.random.rand(N_pts,1,N_pixels_width,N_pixels_width)
    
    #use a random number for each sample point as cutoff
    rand_array[rand_array>np.random.rand(N_pts).reshape(-1,1,1,1)] = 1
    rand_array[rand_array<1] = 0
    
    return rand_array


def random_gen_4(N_pts, N_pixels_width, input_percent_coverage = []):
    """improved random gen with x and y symmetry
    only works for even N_pts
    """
    #determine number of pixels in upper left corner
    N_corner_width = int(N_pixels_width//2 + N_pixels_width%2)
    N_corner_total = N_corner_width*N_corner_width
    
    #initalize array of correct size
    rand_array = np.zeros((N_pts,1,N_corner_width,N_corner_width))
    
    #num of ones is random percentage times number of points in corner
    if input_percent_coverage == []:
        rand_num_ones = np.floor(np.random.rand(N_pts)*N_corner_total).astype(int)
    else:
        rand_num_ones = np.floor(np.array(input_percent_coverage)*N_corner_total).astype(int)
    print("min:",min(rand_num_ones))
    print("max:",max(rand_num_ones))
    
    #repurpose full film and no film to limit values
    rand_num_ones[rand_num_ones==0] = 1
    rand_num_ones[rand_num_ones==N_corner_total] = N_corner_total-1
    
    #populate image with ones based on number of ones needed for each instance
    for i in range(N_pts):
        rand_array_init = np.zeros(N_corner_total)
        rand_array_init[0:rand_num_ones[i]] = 1
        np.random.shuffle(rand_array_init)
        rand_array_init = rand_array_init.reshape(N_corner_width,N_corner_width)
        rand_array[i,0] = rand_array_init
        
    #make symmetrix
    rand_array = np.concatenate((rand_array,np.flip(rand_array,2)),axis=2)
    rand_array = np.concatenate((rand_array,np.flip(rand_array,3)),axis=3)
    
    
    return rand_array

def checkboard(N_pixels_width):
    """generate checkboard design
    """
    #determine number of pixels in upper left corner
    start_pt = [[1+4*i,2+4*j] for i in range(5) for j in range(5)]
    array = np.zeros((2,1,N_pixels_width,N_pixels_width))
    for pt in start_pt:
        array[0,0,pt[0],pt[1]] = 1
        array[0,0,pt[0]+1,pt[1]] = 1
        array[0,0,pt[0],pt[1]-1] = 1
        array[0,0,pt[0]+1,pt[1]-1] = 1

    #add second checkerboard, inverse to first
    #this gives 75% coverage
    array[1,0] = array[0,0]
    array[1,0][array[1,0]==0] = 2
    array[1,0][array[1,0]==1] = 0
    array[1,0][array[1,0]==2] = 1

    return array


def simulated_sym_results(images):
    """simulated results, needs to be finished still!!
    """
    N,C,H,W = images.shape
    target_perc = 0.7
    target_image = random_gen_4(1,W,[target_perc])[0][0]
    plt.imshow(target_image)
    plt.title("simulated ideal image")
    
    loss = abs(images - target_image)
    loss = (loss.sum(axis=(2,3)).reshape(-1))/4 #divide by 4: symmetric
        
    #Tr data converted to ext ratio and dT
    Tr_ins = 1-0.5*loss/max(loss)
    Tr_met = 0.5*loss/max(loss)
    Tr_ins[Tr_ins<0.01] = 0.01
    Tr_met[Tr_met<0.01] = 0.01 

    ext_ratio = 10*np.log10(Tr_ins/Tr_met)
    dT = 10*(1-(loss/max(loss)))
            
    out = np.array((ext_ratio, dT)).transpose()
    out[out==0] = 0.001
    
    labels = 'ext_ratio,dT'
    
    return target_image, labels, out
    

def folder_create(timestamp, master_directory):
    """create folder with timestamp, save readme file
    """
    new_directory = os.path.join(master_directory,timestamp)
    os.mkdir(new_directory)
    print("directory created: ",new_directory)
    return(new_directory)  


def create_timestamp():
    date = datetime.datetime.now()
    timestamp = (str(date.year)[-2:]+ str(date.month).rjust(2,'0')+  
                 str(date.day).rjust(2,'0') 
                 + '-' + str(date.hour).rjust(2,'0') + 
                 str(date.minute).rjust(2,'0'))
    return timestamp


def readme_create(comsol_rev,N,thk):
    """create readme file, save to folder
    """
    readme = 'script name: ' + os.path.basename(__file__) + '\n'
    readme += 'comsol file version = ' + str(comsol_rev) + '\n'
    readme += 'image shape = ' + str(N) + '\n'
    readme += 'thickness = ' + str(thk) + ' (thk[0] is closest to substrate)\n'
    readme += 'additional notes: ' + input('additional notes:  ') + '\n'
    
    return(readme)

##########################################################################
def main():
    image = Image()
    image.input_selection()
    image.create_thk_images()
    
    if input('save output? y to save:  ') == 'y':
        image.save_output()

if __name__ == '__main__':
    main()