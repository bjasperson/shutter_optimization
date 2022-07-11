#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate study matrix and folder of images for pixel optimization
Created on Fri Dec 17 10:28:05 2021

@author: jaspers2
"""

import numpy as np
import datetime
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt


######################################################
#####################Open issues######################
######################################################
# 1) generate seed image
# 2) generate dummy performance data output with selected "optimum" image
    # use binaries: (ideal - current).reshape(-1).sum()
# 3) 


##########################################################################

#master_directory = '/home/jaspers2/Documents/pixel_optimization/validation_data'
#master_directory = '/home/jaspers2/Documents/pixel_optimization/dof_exploration'
master_directory = '/home/jaspers2/Documents/pixel_optimization/prod2_data/cluster'

class Image():
    
    def __init__(self):
        self.thk = [400,240] #thk[0] is closest to substrate
        self.temp = [273.15, 373.15]
        
    def input_selection(self):
        input_list = input('which input list?\n' + 
                           '1=full permutation, generate data\n'+
                           '2=random images, generate data\n'+
                           '3=2 um wide, 0.1 um pixels, 20x20 random symmetric images, generate data\n'+
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
        else:
            raise Exception('Select available input list')
        
    
    
    def blank_feature_image(N_pts,N_thk,num_pixels_width):
        feature_image = np.zeros((N_pts,N_thk,num_pixels_width,num_pixels_width))
        feature_image = np.float32(feature_image)
        return feature_image
    
    def all_permutations(self,N_pixels_width):
        #returns a 2**num_pixels x num_thk x num_pixels_width x num_pixels_width
        N_pixels = N_pixels_width**2
        N_pts = 2**N_pixels
        
        #https://stackoverflow.com/questions/14931769/how-to-get-all-combination-of-n-binary-value
        image_list = list(itertools.product([0,1],repeat = N_pixels))
        image_array = np.array(image_list).reshape(N_pts,1,N_pixels_width,N_pixels_width)
        
        self.images = image_array
        
    def create_thk_images(self):
        thk_array = np.array(self.thk).reshape(len(self.thk),1,1)
        self.images_w_thk = self.images*thk_array
    
    def save_output(self):
        feature_image = self.images_w_thk
        thk = self.thk

        timestamp = create_timestamp()
        
        #make folder
        path = folder_create(timestamp)
        
        ## save image file for NN
        
        # if input_list == '5':
        #     np.save(os.path.join(path,'seed_image'),feature_image)    
            
        #     readme_file = readme_create('na, seed image')
        #     with open(os.path.join(path,'readme.txt'),'w') as output:
        #         output.write(readme_file)
            
        #else:
        np.save(os.path.join(path,timestamp + '_images'),feature_image)
        
        ## save comsol table for Study 1 
        # with open(os.path.join(path,'study1_param_table.txt'),'w') as output:
        #     output.write(study1_table)

        #additional inputs

        N = feature_image.shape

        
        ## save readme
        comsol_rev = input('comsol model revision: ')
        readme_file = readme_create(comsol_rev,N,thk)
        with open(os.path.join(path,'readme.txt'),'w') as output:
            output.write(readme_file)
        
        if input('DEPRECATED - generate squares files for import to comsol (y/n)?     ') == 'y':
            self.save_comsol_inputs(path, timestamp)
            
        if input('generate gen2 param file for cluster+parametric sweep (y/n)?   ') == 'y':
            self.save_comsol_inputs_gen2(path, timestamp)
        
        if input('generate simulated results? y to gen/save:   ') == 'y':
            if input('1) use random training image as ideal or 2) generate new image?    ') == '1':
                target_image_id = int(0.67*self.images.shape[0])
                target_image = feature_image[target_image_id][0]
                print('target_image:\n',target_image)
            else: 
                target_image = random_gen_2(1, self.images.shape[2])[0,0]
                print('target image:\n',target_image)
            
            target_image[target_image>0] = 1
                
            labels, fake_results = simulated_results(self.images,target_image)
            
            
            
            np.savetxt(os.path.join(path,'final_comsol_results.csv'),fake_results,header=labels,comments = '')
            np.save(os.path.join(path,'target_image'),target_image)
        
        
        ## save key
        # key_name = os.path.join(path,'key.csv')
        # key.to_csv(key_name)
       
    def save_comsol_inputs(self, path, timestamp):
        #saves the  file for comsol import        
        path_squares = os.path.join(path,'comsol_import')
        os.mkdir(path_squares)
        
        N,C,H,W = self.images.shape
        
        #make xy grid
        x = np.linspace(-self.cell_width_um/2 + self.pixel_width_um/2,
                        self.cell_width_um/2 - self.pixel_width_um/2,
                        W)
        y = x
        xv, yv = np.meshgrid(x, y)
        xv = xv.reshape(1,1,H,W)
        yv = yv.reshape(1,1,H,W)
        
        x_locs = self.images*xv
        y_locs = self.images*yv
        
        #x_locs = np.trim_zeros(x_locs)
        #y_locs = np.trim_zeros(y_locs)

        for i in range(N):
            x_locs_list = x_locs[i,0][x_locs[i,0] != 0].tolist()
            y_locs_list = y_locs[i,0][y_locs[i,0] != 0].tolist()

            loc_file = ''        
            for j in range(len(x_locs_list)):            
                loc_file += str(x_locs_list[j]) + ',' + str(y_locs_list[j]) + '\n'
            
            with open(os.path.join(path_squares,timestamp+'_'+str(i)+'.csv'),'w') as output:
                output.write(loc_file)
                
        #save paramfile
        paramfile = "filename_start filename_stop base_date base_time\n"
        paramfile += "0 " + str(N-1) + " " + timestamp.split("-")[0] + " " + timestamp.split("-")[1]
        
        with open(os.path.join(path_squares,'paramfile.txt'),'w') as output:
            output.write(paramfile)
            
    def save_comsol_inputs_gen2(self, path, timestamp):
        N,C,H,W = self.images.shape
        
        label = ['bit'+"_"+str(i)+"_"+str(j) for i in range(H) for j in range(W)]
        bits = self.images.reshape(N,-1)
        
        date,time = timestamp.split("-")
        
        #file 1
        # extra_labels = {"base_date":[str(date) for i in range(N)],
        #           "base_time":[str(time) for i in range(N)],
        #           "filename":[str(i) for i in range(N)]}
        
        # temp_273 = {"T_em":["273.15" for i in range(N)]}
        # temp_373 = {"T_em":["373.15" for i in range(N)]}
        

        # #from when I had space separated values for the gui paramtric import
        # df_labels = pd.DataFrame(extra_labels)
        # df_labels = df_labels.transpose()
        
        # df_273 = pd.DataFrame(temp_273)
        # df_273 = df_273.transpose()
        
        # df_373 = pd.DataFrame(temp_373)
        # df_373 = df_373.transpose()
        
        # df = pd.DataFrame(bits.transpose().astype('int'),index=label)
        # df = pd.concat([df,df_labels])
        
        # df_273 = pd.concat([df,df_273])
        # df_373 = pd.concat([df,df_373])
        
        # df_273.to_csv(os.path.join(path,timestamp+'_paramfile_273.txt'),header=False,sep = ",") 
        # df_373.to_csv(os.path.join(path,timestamp+'_paramfile_373.txt'),header=False,sep = ",") 
        ##################
        #file 2
        #from when I was importing directly to the parametric study in the gui
        
        # file = ""
        
        # for i,name in enumerate(label):
        #     file += name + " " + str(bits.transpose()[i].tolist())[1:-1] + "\n"    
            
        
        
        # file += "base_date " + ",".join([str(date) for i in range(N)]) + "\n"
        # file += "base_time " + ",".join([str(time) for i in range(N)]) + "\n"
        # file += "filename " + ",".join([str(i) for i in range(N)]) + "\n"
        
        # file_273 = file + "T_em " + ",".join(["273.15" for i in range(N)])
        # file_373 = file + "T_em " + ",".join(["373.15" for i in range(N)])
        
        
        # with open(os.path.join(path,timestamp+'_paramfile_273_2.txt'),'w') as f:
        #     f.write(file_273)
        
        # with open(os.path.join(path,timestamp+'_paramfile_373_2.txt'),'w') as f:
        #     f.write(file_373)

        ##################
        #file 3
        # paramfile = " ".join(label)
        # paramfile += " base_date base_time filename T_em\n"
        # paramfile_273 = paramfile
        # paramfile_373 = paramfile
        # for i in range(N):
        #     paramfile_273 += np.array2string(bits[i].reshape(-1),max_line_width = 1e6)[1:-1]
        #     paramfile_273 += " " + str(date) + " " + str(time) + " " + str(i) + " " + "273.15\n"
        #     paramfile_373 += np.array2string(bits[i].reshape(-1),max_line_width = 1e6)[1:-1] 
        #     paramfile_373 += " "+ str(date) + " " + str(time) + " " + str(i) + " " + "373.15\n"
            
        
        # with open(os.path.join(path,timestamp+'_paramfile_273_3.txt'),'w') as f:
        #     f.write(paramfile_273)
            
        # with open(os.path.join(path,timestamp+'_paramfile_373_3.txt'),'w') as f:
        #     f.write(paramfile_373)
            
        #bits file
        os.mkdir(os.path.join(path,'bits'))
        for j in range(N):            
            bits_file = ""
            for i,name in enumerate(label):
                bits_file += name + " " + str(bits.transpose()[i,j].tolist()) + "\n"    
            
            with open(os.path.join(path,'bits/'+timestamp+'-'+str(j)+'_bits.txt'),'w') as f:
                f.write(bits_file)
       
        #paramfile    
        # paramfile = "base_date base_time filename_start filename_stop T_em\n"
        # base_string = timestamp.split("-")[0] + " " + timestamp.split("-")[1] + " 0 " + str(N-1) +" "
        # paramfile_273 = paramfile + base_string + "273.15"
        # paramfile_373 = paramfile + base_string + "373.15"
        
        
        # with open(os.path.join(path,'paramfile_273.txt'),'w') as output:
        #     output.write(paramfile_273)
            
        # with open(os.path.join(path,'paramfile_373.txt'),'w') as output:
        #     output.write(paramfile_373)
            

##########################################################################
# def study_table(lists,col_labels):
#     #ref: https://stackoverflow.com/questions/45672342/create-a-dataframe-of-permutations-in-pandas-from-list
#     return pd.DataFrame(list(itertools.product(*lists)), columns=col_labels)

##########################################################################
def random_gen(N_pts, N_pixels_width):
    rand_array = np.random.rand(N_pts,1,N_pixels_width,N_pixels_width)
    rand_array[rand_array>=0.5] = 1
    rand_array[rand_array<0.5] = 0
    
    return rand_array

def random_gen_2(N_pts,N_pixels_width):
    rand_array = np.random.rand(N_pts,1,N_pixels_width,N_pixels_width)
    
    #use a random number for each sample point as cutoff
    rand_array[rand_array>np.random.rand(N_pts).reshape(-1,1,1,1)] = 1
    rand_array[rand_array<1] = 0
    
    return rand_array

def random_gen_3(N_pts,N_pixels_width):
    """random generation with x, y and xy symmetry
    only works for even N_pts for now
    """
    
    #get the upper right
    N_corner = int(N_pixels_width//2 + N_pixels_width%2) 
    rand_array = np.random.rand(N_pts,1,N_corner,N_corner)
    
    #make symmetrix
    rand_array = (rand_array + rand_array.transpose(0,1,3,2))/2
    rand_array = np.concatenate((rand_array,np.flip(rand_array,2)),axis=2)
    rand_array = np.concatenate((rand_array,np.flip(rand_array,3)),axis=3)
    
    #use a random number for each sample point as cutoff

    ##### attempt to make sure cutoff is less than largest number    
    # for i in range(N_pts):
    #     cutoff = np.random.rand(1).reshape(1,1)
            
    #     current_image = rand_array[i,0]
        
    #     while np.all(current_image>cutoff):
    #         cutoff = np.random.rand(1).reshape(1,1)
        
    #     rand_cutoff_array = cutoff.reshape(1,1,1,1)
    # rand_cutoff_array = cutoff.reshape(1,1,1,1)
    
        
    ###Legacy, working code
    rand_cutoff_array = np.random.rand(N_pts).reshape(-1,1,1,1)
    rand_array[rand_array>rand_cutoff_array] = 1
    rand_array[rand_array<1] = 0
    
    return rand_array

def random_gen_4(N_pts,N_pixels_width):
    """improved random gen with x, y and xy symmetry
    only works for even N_pts for now
    """
    N_corner_width = int(N_pixels_width//2 + N_pixels_width%2)
    N_corner_total = N_corner_width*N_corner_width
    
    rand_array = np.zeros((N_pts,1,N_corner_width,N_corner_width))
    rand_num_ones = np.floor(np.random.rand(N_pts)*N_corner_total).astype(int)
    print("min:",min(rand_num_ones))
    print("max:",max(rand_num_ones))
    rand_num_ones[rand_num_ones==0] = 1
    rand_num_ones[rand_num_ones==N_corner_total] = N_corner_total-1
    
    for i in range(N_pts):
        rand_array_init = np.zeros(N_corner_total)
        rand_array_init[0:rand_num_ones[i]] = 1
        np.random.shuffle(rand_array_init)
        rand_array_init = rand_array_init.reshape(N_corner_width,N_corner_width)
        rand_array[i,0] = rand_array_init
        
    #make symmetrix
    rand_array = np.concatenate((rand_array,np.flip(rand_array,2)),axis=2)
    rand_array = np.concatenate((rand_array,np.flip(rand_array,3)),axis=3)
    
    
    return(rand_array)

##########################################################################
def simulated_results(images,ideal_image):
    loss = abs(images - ideal_image)
    loss = loss.sum(axis=(2,3)).reshape(-1)
    Tr_ins = 1-loss/max(loss)
    Tr_met = loss/max(loss)
    R_ins = (1 - Tr_ins)*0.5
    R_met = (1 - Tr_met)*0.5
    A_ins = 1 - Tr_ins - R_ins
    A_met = 1 - Tr_met - R_met
    Temp = 340-70*loss/max(loss)
    
    out = np.array((R_ins,R_met,Tr_ins,Tr_met,A_ins,A_met,Temp)).transpose()
    out[out==0] = 0.001
    
    labels = 'R_ins R_met Tr_ins Tr_met A_ins A_met Temp'
    
    return labels, out

def folder_create(timestamp):
    #create folder with timestamp, save readme file
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

##########################################################################
def readme_create(comsol_rev,N,thk):
    #create readme file, save to folder
    readme = 'script name: ' + os.path.basename(__file__) + '\n'
    readme += 'comsol file version = ' + str(comsol_rev) + '\n'
    readme += 'image shape = ' + str(N) + '\n'
    readme += 'thickness = ' + str(thk) + ' (thk[0] is closest to substrate)\n'
    readme += 'additional notes: ' + input('additional notes:  ') + '\n'
    
    # readme += '--------------------------------\n'
    # readme += 'study1_output table includes: lda0, eps_VO2_avg, R, Tr, A, Total, T_VO2_avg\n'
    # readme += 'study2_output table includes: T_VO2_avg, A_applied_avg\n'

    # readme += '--------------------------------'
    
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