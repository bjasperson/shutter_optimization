import pixel_optim_nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import pickle
import glob
from image_creation import create_timestamp
import os

def generate(save_folder, dB_range = (2,15), n_dB = 50):
    dB_values = np.linspace(dB_range[0],dB_range[1],n_dB).tolist()
    data_folder = '/home/jaspers2/Documents/pixel_optimization/prod3_data/combined_results_221004-1015_withdT/trained_model_221004-1433'
    images, target_dB_list, pred_dB_list, pred_dT_list, path_list,bits_list = [],[],[],[],[],[]
    unique_id_list = []

    timestamp = create_timestamp()
    
    for i,dB in enumerate(dB_values):
        top_opt_current = pixel_optim_nn.top_opt_funct(data_folder,
                                            target_db = dB)
        path = top_opt_current.save_results(save_folder, return_directory = True)
        
        #get bits
        bits_path = path + '/bits'
        for filename in glob.iglob(f'{bits_path}/*'):
            with (open(filename,'rt')) as openfile:
                bits = openfile.read()
        
        images.append(top_opt_current.images.detach().numpy()[0][0])
        pred_dB, pred_dT = top_opt_current.predicted_perf.labels.tolist()[0]
        target_dB_list.append(dB)
        pred_dB_list.append(pred_dB)
        pred_dT_list.append(pred_dT)
        path_list.append(path)
        bits_list.append(bits)
        unique_id_list.append(timestamp + '-' + str(i))
    
    data = {'images':images,
            'target_dB':target_dB_list,
            'pred_dB':pred_dB_list,
            'pred_dT':pred_dT_list,
            'path':path_list,
            'bits':bits_list,
            'id':unique_id_list}
    
    df = pd.DataFrame(data)
    df.to_pickle(save_folder+'/dB_study_data.pkl')
    return df


def load_nn_data(data_folder = ''):
    if data_folder == '':
        data_folder = input('Folder w/ pkl file:')
    file = data_folder + '/dB_study_data.pkl'
    with (open(file,'rb')) as openfile:
        df = pickle.load(openfile)
    return df


def load_comsol_results(df_nn, folder = ''):
    """combines text file results from results folder, returns df
    """
    if folder == '':
        folder = input('folder of results: ')
    df = pd.DataFrame()
    for file in os.listdir(folder):
        df_in = pd.read_csv(os.path.join(folder,file),index_col='id')
        
        df = pd.concat([df,df_in])
        df = df.groupby("id").first()

    df["ext_ratio"] = 10*np.log10(df["Tr_ins"]/df["Tr_met"])
    df["insert_loss"] = 10*np.log10(1/df["Tr_ins"])
    df["dT"] = df["T_VO2_avg"]-273.15
    #df = df.set_index("id") 

    #legacy issue
    if 'unique_id' in df_nn.columns:
        df_nn['id'] = df_nn['unique_id']

    df_nn = df_nn.set_index('id')

    combined = pd.concat([df_nn,df],axis=1)

    return combined


def plotting(df, n_plt_w = 3, n_plt_h = 3):
    #subplots of select designs
    i=0
    j=0
    # pred_dB_list = []
    # pred_dT_list = []
    fig,axs = plt.subplots(n_plt_h,n_plt_w)    
    n_plt = n_plt_w*n_plt_h

    n_df = df.shape[0]
    data_pt = [round(i*(n_df/(n_plt-1)))for i in range(n_plt-1)]
    data_pt.append(n_df-1)
    for p in data_pt:
        current = df.iloc[p]
        image = current.images
        image = np.concatenate((image,np.flip(image,0)),axis=0)
        image = np.concatenate((image,np.flip(image,1)),axis=1)
        axs[i,j].imshow(image)
        axs[i,j].get_xaxis().set_visible(False)
        axs[i,j].get_yaxis().set_visible(False)
        axs[i,j].set_title(f"{p}: dT={current.pred_dT:.1f}, dB={current.pred_dB:.1f} ({current.target_dB:.1f})", fontsize=8)

        if j<(n_plt_w-1):
            j+=1
        else:
            i+=1
            j=0


    #create the ext_ratio vs dT plot, with pareto front
    file_path = "/home/jaspers2/Documents/pixel_optimization/prod3_data/combined_results_221004-1015_withdT"
    df_comsol = pd.read_pickle(file_path+"/df_all.pkl")
    df_comsol["dT"] = df_comsol["T_VO2_avg"] - 273.15
    df_opt = df[['ext_ratio','dT','pred_dB','pred_dT']]
    #df_opt = df_opt.rename(columns={'pred_dB':'ext_ratio','pred_dT':'dT'})
    
    #drop nas for now
    df_opt = df_opt.dropna()

    plt.figure()
    plt.scatter(df_comsol["ext_ratio"],
                df_comsol["dT"],
                label='Training',
                s=5)
    plt.scatter(df_opt["ext_ratio"],
                df_opt["dT"],
                marker='o',
                label='COMSOL Verified')
    plt.scatter(df_opt["pred_dB"],
                df_opt["pred_dT"],
                marker='x',
                label='NN Prediction')
    #plt.title("Training Data")
    plt.xlabel(r"Extinction Ratio [dB] = $10\log_{10}\frac{Tr_{ins}}{Tr_{met}}$")
    plt.ylabel("Temperature Rise - K")
    plt.legend()
    plt.grid()
    plt.show()

   
def save_pdf(df, folder):

    with PdfPages(folder+'/all_images.pdf') as pdf:
        for p,current in df.iterrows():
            image = current.images
            image = np.concatenate((image,np.flip(image,0)),axis=0)
            image = np.concatenate((image,np.flip(image,1)),axis=1)
            fig,axs = plt.subplots(1,1)   
            axs.imshow(image)
            axs.get_xaxis().set_visible(False)
            axs.get_yaxis().set_visible(False)
            axs.set_title(f"{p}\ndT (Pred (NN), FEM)= {current.dT:.1f}, {current.pred_dT:.1f}\ndB (Target, Pred (NN), FEM)  = {current.target_dB:.1f}, {current.pred_dB:.1f}, {current.ext_ratio:.1f}", fontsize=8)
            pdf.savefig()
            plt.close(fig)

    return


def save_bits(df, save_folder):
    for p,current in df.iterrows():
        bits_str = df['bits'].loc[p]
        name = df['id'].loc[p]
        filename = save_folder + '/all_bits/' + name +'_bits.txt'
        with open(filename,"w") as text_file:
            text_file.write(bits_str)

def main():
    base_folder = '/home/jaspers2/Documents/pixel_optimization/dB_study/230225-1625'
    df = load_nn_data(base_folder)
    df = load_comsol_results(df, base_folder + '/results')
    plotting(df)
    save_pdf(df, base_folder)
    return df

if __name__ == '__main__':
    df = main()