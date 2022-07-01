#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 06:21:55 2022

@author: jaspers2
"""

import numpy as np
import pandas as pd
import os
import csv



def main():
    base_folder = "/home/jaspers2/Documents/pixel_optimization/prod1_data/"
    image_folder = base_folder + "images" 
    results_folder = base_folder + "results"
    
    #import all files in folder
    df_images = import_images(image_folder)
    df_results = import_results(results_folder)
    
    df_combined = combined_results(df_images,df_results)
    np_combined_images,df_combined_results = nn_config(df_combined)
    save_results(base_folder, np_combined_images, df_combined_results)
    
    pass


def import_images(folder):
    
    df = pd.DataFrame()
    for file in os.listdir(folder):
        images_in = np.load(os.path.join(folder,file))
        
        N,C,H,W = images_in.shape
        
        datecode = file.split("_")[0]
        labels_in = [datecode+"-"+str(i) for i in range(N)]
        df_in = pd.DataFrame({"id":labels_in,"image":[image for image in images_in]})
        df_in.set_index('id')
        df = pd.concat([df,df_in])
        
    df = df.set_index('id')
    return df

def import_results(folder):
    #returns df of results from text files in folder
    df = pd.DataFrame()
    for file in os.listdir(folder):
        df_in = pd.read_csv(os.path.join(folder,file),index_col='id')
        
        df = pd.concat([df,df_in])
        df = df.groupby("id").first()
    
    return(df)

def analyze_results():
    #perform any analysis on results as needed
    pass

def combined_results(images,results):
    #this should call import_images and import_results
    #then, sort and combine into format needed by pixel_nn.py
    combined = pd.concat([images,results],axis=1).dropna(axis=0)    
    
    return(combined)

def nn_config(combined_df):
    #saves the combined feature_images.npy
    np_images = combined_df['image'].to_numpy()
    np_images = [np_images[i] for i in range(len(np_images))]
    np_images = np.asarray(np_images)
    
    #saves the combined csv: R_ins,R_met,Tr_ins,Tr_met,A_ins,A_met,Temp
    np_results = combined_df[['R_ins','R_met','Tr_ins','Tr_met','A_ins','A_met','T_VO2_avg']]
    np_results = np_results.rename(columns={'T_VO2_avg':'Temp'})
    
    return(np_images,np_results)

def save_results(path, np_images,np_results):
    
    np.save(os.path.join(path,'feature_images'),np_images)
    np_results.to_csv(path + 'final_comsol_results.csv', index=False)


if __name__ == '__main__':
    main()