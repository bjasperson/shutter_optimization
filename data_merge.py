#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 06:21:55 2022

@author: jaspers2
"""

import numpy as np
import pandas as pd
import os
import image_creation
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200

def main():    
    #process input data, merge and create combined results
    if input("process input data (y/n)? ")=="y":
        readme = ""
        base_folder = input("prod[x]_data folder:  ")
        image_folder = base_folder + "/images" 
        results_folder = base_folder + "/results"
        
        #timestamp/make directory
        timestamp = image_creation.create_timestamp()
        combined_results_folder = base_folder + "/combined_results_" + timestamp + "/"
        os.mkdir(combined_results_folder)
        
        #import all files in folder
        df_images = import_images(image_folder)
        df_results = import_results(results_folder)
        
        #
        df_combined = combined_results(df_images,df_results)
        df_combined, np_combined_images, np_reduced_images, df_combined_results, readme = nn_config(df_combined,readme)
        save_results(combined_results_folder, df_combined, np_combined_images, np_reduced_images, df_combined_results, readme)
    
    if input("analyze results (y/n)?  ")=="y":
        check = "combined_results_folder" in locals()
        if check == False:
            combined_results_folder = input("combined results folder: ")
        analyze_results(combined_results_folder)  
        

def import_images(folder):
    """import image files
    """
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
    """combines text file results from results folder, returns df
    """
    
    #try this next: https://python-bloggers.com/2021/09/3-ways-to-read-multiple-csv-files-for-loop-map-list-comprehension/
    #read tarfiles directly: https://stackoverflow.com/questions/2018512/reading-tar-file-contents-without-untarring-it-in-python-script
    
    df = pd.DataFrame()
    for file in os.listdir(folder):
        df_in = pd.read_csv(os.path.join(folder,file),index_col='id')
        
        df = pd.concat([df,df_in])
        df = df.groupby("id").first()
    
    return df

def analyze_results(file_path):
    """perform any analysis on results as needed
    """
    df = pd.read_pickle(file_path+"/df_all.pkl")
    df["dT"] = df["T_VO2_avg"] - 273.15
    df.plot('dT','q_applied',kind="scatter")
    df.plot("ext_ratio","dT",kind="scatter",xlabel="Extinction Ratio - dB",ylabel="Temperature rise - K",grid=True)
    df.plot("ext_ratio","insert_loss",kind="scatter",xlabel="Extinction Ratio - dB",ylabel="Insertion Loss -dB",grid=True)
    df.plot("insert_loss","dT",kind="scatter",xlabel="Insertion Loss - dB",ylabel="Temperature rise -K",grid=True)
    df.hist("ext_ratio")
    df.hist("insert_loss")

    

    
    plt.figure()
    plt.scatter(df["ext_ratio"],
             df["dT"])
    plt.title("Training Data")
    plt.xlabel("Extinction Ratio - dB")
    plt.ylabel("Temperature Rise - K")
    plt.grid()
    plt.show()
    
    print(df[["T_VO2_avg","ext_ratio","insert_loss"]].sort_values("ext_ratio"))
    print(df[df['T_VO2_avg']>281.65][['T_VO2_avg','q_applied']].sort_values(['T_VO2_avg','q_applied']))
    
    coverage = []
    for i,image in enumerate(df["image"]):
        total = sum(image[0].reshape(-1)>0)
        coverage.append(total)
    df["coverage"] = coverage/max(coverage)

    df.plot("coverage","ext_ratio",kind="scatter",ylabel="Extinction Ratio - dB",grid=True)
    df.plot("coverage","insert_loss",kind="scatter",ylabel="Insertion Loss - dB",grid=True)
    df.plot("coverage","dT",kind="scatter",ylabel="Temperature Rise -K",grid=True)
    

def combined_results(images,results):
    """creates one df with images and results synced up
    """
    combined = pd.concat([images,results],axis=1).dropna(axis=0)    
    combined["ext_ratio"] = 10*np.log10(combined["Tr_ins"]/combined["Tr_met"])
    combined["insert_loss"] = 10*np.log10(1/combined["Tr_ins"])
    
    return combined


def nn_config(combined_df,readme):
    """saves the combined feature_images.npy
    """
    
    #temporarily remove any Tr_ins < 0.5 and ext_ratio < 0
    if input("remove Tr_ins<0.5 and ext_ratio<0? y/n:  ")=="y":
        combined_df = combined_df[combined_df["ext_ratio"]>0]
        combined_df = combined_df[combined_df["Tr_ins"]>0.5]
        readme += 'removed Tr_ins < 0.5 and ext_ratio < 0\n'
        
    if input("remove Tr_met>0.2? y/n:  ")=="y":
        combined_df = combined_df[combined_df["Tr_met"]<0.2]
        readme += 'removed Tr_met > 0.2\n'
    
    
    #clean up for numpy array
    np_images = combined_df['image'].to_numpy()
    np_images = [np_images[i] for i in range(len(np_images))]
    np_images = np.asarray(np_images)
    
    #created reduced images (leverage symmetry)
    readme += "feature_images.py = upper left corner, full_feature_images.npy = full symmetric image\n"
    N,C,H,W = np_images.shape
    np_reduced_images = np_images[:,:,:H//2,:W//2]
    
    #saves the combined csv: R_ins,R_met,Tr_ins,Tr_met,A_ins,A_met,Temp
    df_results = combined_df[['ext_ratio','insert_loss','T_VO2_avg']]
    df_results = df_results.rename(columns={'T_VO2_avg':'Temp'})
    
    return combined_df,np_images,np_reduced_images,df_results,readme


def save_results(path, df_all, np_images, np_reduced_images, df_results, readme):
    """saves all results
    """
    df_all.to_pickle(path + 'df_all.pkl')
    np.save(path + 'full_feature_images',np_images)
    np.save(path + 'feature_images',np_reduced_images)
    df_results.to_csv(path + 'final_comsol_results.csv', columns=['ext_ratio','insert_loss','Temp'], index=False)
    
    with open(os.path.join(path,'readme.txt'),'w') as output:
        output.write(readme)


if __name__ == '__main__':
    main()