import pixel_optim_nn
import pixel_nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import pickle
import glob
from image_creation import create_timestamp
import os
import torch

def generate(save_folder, dB_range = (2,15), n_dB = 100, p_max_in = 2):
    dB_values = np.linspace(dB_range[0],dB_range[1],n_dB).tolist()
    data_folder = './data/combined_results_dT/trained_model_221004-1433'
    images, images_tf, target_dB_list, pred_dB_list, pred_dT_list, path_list,bits_list = [],[],[],[],[],[],[]
    unique_id_list = []

    timestamp = create_timestamp()
    
    for i,dB in enumerate(dB_values):
        print(f'running dB={dB}')
        top_opt_current = pixel_optim_nn.top_opt_funct(data_folder,
                                                       target_db = dB,
                                                       num_epochs = 3000,
                                                       p_max = p_max_in,
                                                       print_details = False)
        path = top_opt_current.save_results(save_folder, return_directory = True)
        
        #get bits
        bits_path = path + '/bits'
        for filename in glob.iglob(f'{bits_path}/*'):
            with (open(filename,'rt')) as openfile:
                bits = openfile.read()
        
        images.append(top_opt_current.images.detach().numpy()[0][0])
        images_tf.append(top_opt_current.images.detach().numpy())
        pred_dB, pred_dT = top_opt_current.predicted_perf.labels.tolist()[0]
        target_dB_list.append(dB)
        pred_dB_list.append(pred_dB)
        pred_dT_list.append(pred_dT)
        path_list.append(path)
        bits_list.append(bits)
        unique_id_list.append(timestamp + '-' + str(i))
    
    data = {'images':images,
            'images_tf':images_tf,
            'target_dB':target_dB_list,
            'pred_dB':pred_dB_list,
            'pred_dT':pred_dT_list,
            'path':path_list,
            'bits':bits_list,
            'id':unique_id_list}
    
    df = pd.DataFrame(data)
    df.to_pickle(save_folder+'/dB_study_data.pkl')
    return df


def simp_p_study(df = '', save_results = False):
    
    if len(df) == 0:
        p_range_max = 6 
        p_range_min = 1
        p_step = .5
        num_it = 15
        num_epochs = 3000
        p_range = np.arange(p_range_min,p_range_max,p_step)

        target_ext_ratio = 10

        ext_ratio_error = []
        study_details = []
        rho_converge_perc = []
        for p_max_in in p_range:
            pred_ext_ratio = []
            interm_rho = []
            for i in range(num_it):
                print(f'running: p={p_max_in}, run {i}')
                top_opt_current = pixel_optim_nn.top_opt_funct(trained_model_folder,
                                                        target_db = target_ext_ratio,
                                                        num_epochs = num_epochs,
                                                        p_max = p_max_in,
                                                        p_step = (p_max_in-1)/num_epochs,
                                                        print_details = False)
                er = top_opt_current.predicted_perf.labels.detach().tolist()[0][0]
                rho = top_opt_current.top_net.rho.detach().reshape(-1)
                not_converged = len(rho[(rho > 0.1) & (rho < 0.9)])/len(rho)
                interm_rho.append(not_converged)
                pred_ext_ratio.append(er)
                print('-----')
            avg_ext_ratio = np.array(pred_ext_ratio).mean()
            std_ext_ratio = np.array(pred_ext_ratio).std()
            ext_ratio_error.append(abs(avg_ext_ratio-target_ext_ratio))
            rho_converge_perc.append(np.array(interm_rho).mean())
            study_details.append({'p_range':(p_range_min,p_range_max),
                                  'pred_ext_ratio_stdev':std_ext_ratio,
                                  'num_iterations':num_it,
                                  'target_ext_ratio':target_ext_ratio,
                                  'ext_ratio_list':pred_ext_ratio,
                                  'not_converged':interm_rho})
        df = pd.DataFrame({'p_max':p_range,
                           'abs_ext_ratio_error':ext_ratio_error,
                           'rho_converge_perc':rho_converge_perc,
                           'study_details':study_details,
                           })

    fig,ax = plt.subplots()
    ax.plot(df['p_max'], df['abs_ext_ratio_error'])
    ax.set_xlabel("maximum p setting")
    ax.set_ylabel("abs(Predicted - Target) Ext. Ratio")
    fig.show()
    if save_results == True:
        fig.savefig(os.path.expanduser("./studies/p_study/p_study.png"))
        df.to_csv(os.path.expanduser("./studies/p_study/p_study.csv"),index=False)



    return df

def adam_learning_rate_study():
    data_dir_in = "./data/combined_results_dT"
    #learning_rate_list = np.arange(1e-3,3e-3,1e-2)
    n_list = np.arange(2.,6.,1.)
    learning_rate_list, out = [], []
    for n in n_list:
        learning_rate_in = 10**(-n)
        current = pixel_nn.main(data_dir = data_dir_in,
                      learning_rate = learning_rate_in,
                      save_out=False)
        perc_db_error = (current.error.mean(axis=0)[0])*100
        out.append(perc_db_error)
        learning_rate_list.append(learning_rate_in)
    
    fig,ax = plt.subplots()
    ax.semilogx(learning_rate_list,out)
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("perc Ext. Ratio error")
    fig.show()
    return out, learning_rate_list


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
    df_c = pd.DataFrame()
    for file in os.listdir(folder):
        df_in = pd.read_csv(os.path.join(folder,file),index_col='id')
        
        df_c = pd.concat([df_c,df_in])
        df_c = df_c.groupby("id").first()

    df_c["ext_ratio"] = 10*np.log10(df_c["Tr_ins"]/df_c["Tr_met"])
    df_c["insert_loss"] = 10*np.log10(1/df_c["Tr_ins"])
    df_c["dT"] = df_c["T_VO2_avg"]-273.15
    #df = df.set_index("id") 

    #legacy issue
    if 'unique_id' in df_nn.columns:
        df_nn['id'] = df_nn['unique_id']

    df_nn = df_nn.set_index('id')

    combined = pd.concat([df_nn,df_c],axis=1)

    return combined


def plot_extRatio_vs_dT(df, filtering = False,
                        save_fig = False):
    #create the ext_ratio vs dT plot, with pareto front
    file_path = "./data/combined_results_dT"
    df_comsol = pd.read_pickle(file_path+"/df_all.pkl")
    df_comsol["dT"] = df_comsol["T_VO2_avg"] - 273.15
    
    col_names = ['pred_dB','pred_dT']
    if 'ext_ratio' in df.columns:
        col_names.extend(['ext_ratio','dT'])
    
    df_opt = df[col_names]
    #df_opt = df_opt.rename(columns={'pred_dB':'ext_ratio','pred_dT':'dT'})
    
    #drop nas for now
    df_opt = df_opt.dropna()

    fig,ax = plt.subplots()
    #plt.figure()

    if 'ext_ratio' in df_opt.columns:
        ax.scatter(df_comsol["ext_ratio"],
                    df_comsol["dT"],
                    label='Training',
                    s=5)
    
    if "ext_ratio" in df_opt.columns:
        ax.scatter(df_opt["ext_ratio"],
                    df_opt["dT"],
                    marker='o',
                    label='COMSOL Verified')
    ax.scatter(df_opt["pred_dB"],
                df_opt["pred_dT"],
                marker='x',
                label='NN Prediction')
    #ax.set_title("Training Data vs. TopOptNN output")
    ax.set_xlabel(r"Extinction Ratio [dB] = $10\log_{10}\frac{Tr_{ins}}{Tr_{met}}$")
    ax.set_ylabel("Temperature Rise - K")
    ax.legend()
    ax.grid(True)
    plt.show()
    if save_fig == True:
        fig.savefig(os.path.expanduser("./figs/FIG10a_dB.eps"),bbox_inches = "tight")

def plot_best_comsol(df,db_min, db_max, filtering=False):

    
    df_filtered = df.loc[(df['ext_ratio'] > db_min) & (df['ext_ratio'] < db_max)]
    df_best = df_filtered[df_filtered['dT'] == df_filtered['dT'].max()]
    image = df_best['images'][0]
    image = np.concatenate((image,np.flip(image,0)),axis=0)
    image = np.concatenate((image,np.flip(image,1)),axis=1)
    if filtering == True:
        image[image<0.5] = 0
        image[image>=0.5] = 1
    fig,ax = plt.subplots()
    ax.set_title(f"Max Temperature Candidate, {db_min} to {db_max} dB")
    ax.imshow(image)

    return ax

def isolate_best_comsol(df, n_best_pts, db_range = []):
    df = df.reset_index()
    if db_range == []:
        db_min = df.ext_ratio.min()
        db_max = df.ext_ratio.max()
    else:
        db_min = db_range[0]
        db_max = db_range[1]
    db_range_array = np.linspace(db_min, db_max, n_best_pts)
    df_pareto = []

    for k in range(n_best_pts-1):
        db_target_min = db_range_array[k]
        db_target_max = db_range_array[k+1]
        df_filtered = (df.loc[(df['ext_ratio'] > db_target_min) & (df['ext_ratio'] < db_target_max)])
        df_best = df_filtered[df_filtered['dT'] == df_filtered['dT'].max()]
        if len(df) == 0:
            raise Exception('distance b/w target DBs is too small to find a datapoint')
        df_pareto.append(df_best.to_dict(orient='records')[0])

    df_pareto = pd.DataFrame(df_pareto)
    return df_pareto


def plot_best_comsol_designs(df, db_ranges, filtering = False, save_fig = False):
    n_plts = len(db_ranges)-1
    n_plt_h = round(n_plts**.5)
    n_plt_w = n_plts//n_plt_h + (n_plts%n_plt_h > 0)
    fig,axs = plt.subplots(n_plt_h, n_plt_w, figsize=(6,6))
    i,j=0,0
    for k in range(len(db_ranges)-1):
        
        db_min = db_ranges[k]
        db_max = db_ranges[k+1]
        
        df_filtered = df.loc[(df['ext_ratio'] > db_min) & (df['ext_ratio'] < db_max)]
        df_best = df_filtered[df_filtered['dT'] == df_filtered['dT'].max()]
        image = df_best['images'][0]
        image = np.concatenate((image,np.flip(image,0)),axis=0)
        image = np.concatenate((image,np.flip(image,1)),axis=1)
        if filtering == True:
            image[image<0.5] = 0
            image[image>=0.5] = 1
        axs[i,j].set_title(f"{float(df_best['ext_ratio']):.1f} dB, {float(df_best['dT']):.1f} K")
        axs[i,j].get_xaxis().set_visible(False)
        axs[i,j].get_yaxis().set_visible(False)
        axs[i,j].imshow(image)

        if j<(n_plt_w-1):
            j+=1
        else:
            i+=1
            j=0

    #fig.suptitle('Best Performing Designs (FEM results)')
    fig.show()

    if save_fig == True:
        fig.savefig(os.path.expanduser("./figs/FIG10b_best.eps"),bbox_inches = "tight")

    return 


def save_pdf(df, folder, filtered = False):
    
    if filtered == True:
        name = folder+'/all_images_filtered.pdf'
    elif filtered == False:
        name = folder+'/all_images.pdf'

    with PdfPages(name) as pdf:
        for p,current in df.iterrows():
            image = current.images
            image = np.concatenate((image,np.flip(image,0)),axis=0)
            image = np.concatenate((image,np.flip(image,1)),axis=1)
            if filtered == True:
                image[image<0.5] = 0
                image[image>=0.5] = 1
            fig,axs = plt.subplots(1,1)   
            axs.imshow(image)
            axs.get_xaxis().set_visible(False)
            axs.get_yaxis().set_visible(False)
            axs.set_title(f"{p}\ndT (Pred (NN), FEM)= {current.pred_dT:.1f}, {current.dT:.1f}\ndB (Target, Pred (NN), FEM)  = {current.target_dB:.1f}, {current.pred_dB:.1f}, {current.ext_ratio:.1f}", fontsize=8)
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


def get_prediction(np_image, perfnet):

    pred_tf = perfnet(torch.tensor(np_image))

    pred = pixel_optim_nn.Labels(perfnet)
    pred.label_update(pred_tf,'normalized')
    pred.scale_labels()
    
    return pred.labels.tolist()

def add_filtered_image_perf(df, perfnn):
    pred_dB_filtered, pred_dT_filtered = [],[]
    for i,row in df.iterrows():
        filtered_image = row['images_tf']
        filtered_image[filtered_image<0.5] = 0
        filtered_image[filtered_image>=0.5] = 1
        pred_filtered = get_prediction(filtered_image, perfnn)[0]
        pred_dB_filtered.append(pred_filtered[0])
        pred_dT_filtered.append(pred_filtered[1])

    df['pred_dB_filtered_image'] = pred_dB_filtered
    df['pred_dT_filtered_image'] = pred_dT_filtered
        
    return df

def add_error(df):
    df['nn_db_error'] = abs(df['target_dB'] - df['pred_dB'])
    df['nn_comsol_error'] = abs(df['target_dB'] - df['ext_ratio'])
    df['nn_dT_error'] = df['pred_dT'] - df['dT']

    fig,ax = plt.subplots()
    ax.plot(df['target_dB'],df['nn_db_error'],label='NN')
    ax.plot(df['target_dB'],df['nn_comsol_error'],label='COMSOL')
    ax.set_xlabel('Target Ext. Ratio [dB]')
    ax.set_ylabel('Absolute Error, Predicted Ext. Ratio [dB]')
    ax.set_title(f'FEM Mean Error, Stdev: {df.nn_comsol_error.mean():.3f}, {df.nn_comsol_error.std():.3f}')
    fig.legend()
    plt.show()

    fig,ax = plt.subplots()
    ax.scatter(df['target_dB'],df['nn_dT_error'])
    ax.set_xlabel('Target Ext. Ratio [dB]')
    ax.set_ylabel('dT Error, NN-FEM [K]')
    fig.legend()
    plt.show()


    return df

def db_study_no_comsol(base_folder, save = False):
    df = load_nn_data(base_folder)
    #df = add_error(df)
    if save == True:
        save_pdf(df, base_folder)
    #db_ranges = [0,2,4,6,8,10,12,15,16,17]
    model = pixel_optim_nn.load_perfnet(trained_model_folder)
    df = add_filtered_image_perf(df, model)
    plot_extRatio_vs_dT(df)
    #df_pareto = isolate_best_comsol(df, 10)

    return df, model

def db_study_load_and_plot(base_folder, save = False):    
    df = load_nn_data(base_folder)
    df = load_comsol_results(df, base_folder + '/results')
    df = add_error(df)
    if save == True:
        save_pdf(df, base_folder)
        save_pdf(df, base_folder, filtered = True)
    #db_ranges = [0,4,6,8,10,12,14,18]
    db_ranges = [0,2,4,6,8,10,12,14,17.5,20] #for paper
    plot_best_comsol_designs(df, db_ranges, filtering = True, save_fig = save)
    model = pixel_optim_nn.load_perfnet(trained_model_folder)
    df = add_filtered_image_perf(df, model)
    df_pareto = isolate_best_comsol(df, 10)
    plot_extRatio_vs_dT(df, save_fig = save)

    return df, df_pareto, model

def generate_for_manuscript(save = True):
    base_folder = os.path.expanduser("./studies/db_ranges/manuscript")
    df, df_pareto, model = db_study_load_and_plot(base_folder, save = save)
    return df, df_pareto, model

if __name__ == '__main__':
    trained_model_folder = "./data/combined_results_dT/trained_model_221004-1433"
    generate_for_manuscript()