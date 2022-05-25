from re import A
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib import cm
import os

marker_dict = {
    'TwoSampleConfounderTest'       : 'x',
    'FullTwoSampleConfounderTest'   : 's',
    'EnvironmentTest'               : 'd',
    'PermutationConfounderTest'     : 'o'
}

def set_mpl_default_settings():
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']) # Set the default color cycle
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.size'] = 18


def plot_experiment_results(df : pd.DataFrame, experiment_desc = None, vary_lambda_only = False):
    
    plt.figure() 
    cmap = plt.cm.get_cmap('YlGnBu')
    
    # List of points in x axis
    XPoints     = [] # X var

    # List of points in y axis
    YPoints     = [] # Y var

    if not vary_lambda_only:
        Ykey = 'confounder_strength'
    else:
        Ykey ='X_beta'

    for Xval in df.nbr_env.unique():
        XPoints.append(Xval)
    for Yval in df[Ykey].unique():
        YPoints.append(Yval)
    
    XPoints.sort()
    YPoints.sort()

    # Z values as a matrix
    ZPoints     = np.ndarray((len(YPoints), len(XPoints)))

    # Populate Z Values 
    for x in range(0, len(XPoints)):
        for y in range(0, len(YPoints)):
            
            filter1 = (np.abs(df.nbr_env - XPoints[x]) < 1e-3)
            filter2 = (np.abs(df[Ykey] - YPoints[y]) < 1e-3)

            tmp_df = df[filter1 & filter2]

            assert len(tmp_df) == 1, 'There are more than one value for the reject rate'
                        
            ZPoints[y][x] = tmp_df.reject_rate.values[0] # Assumes only one value per x,y combination
    print(ZPoints)
    contours = plt.contourf(XPoints, YPoints, ZPoints, cmap=cmap, vmin=0, vmax=1, extent='both')

    # Labels
    plt.xlabel('Number of environments $K$')
    plt.ylabel('Confounder effect size $\gamma$')

    # Colorbar
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap))
    cbar.set_label('Probability of detection')

    if experiment_desc:
        path = os.path.join('results/figures', experiment_desc)
        plt.savefig(path+'.pdf', format='pdf', bbox_inches='tight')

def contour_plot(df : pd.DataFrame, nbr_samples, nbr_env, conf_strength):

    plt.figure()

    # List of points in x axis
    XPoints     = [] # X var

    # List of points in y axis
    TPoints     = [] # T var

    # X and Y points are from -6 to +6 varying in steps of 2 
    for Xval in df.X_b.unique():
        XPoints.append(Xval)
    for Tval in df.T_b.unique():
        TPoints.append(Tval)

    XPoints.sort()
    TPoints.sort()

    # Z values as a matrix
    ZPoints     = np.ndarray((len(TPoints), len(XPoints)))

    # Populate Z Values (a 7x7 matrix) - For a circle x^2+y^2=z    
    for x in range(0, len(XPoints)):
        for t in range(0, len(TPoints)):
            
            filter1 = (np.abs(df.X_b - XPoints[x]) < 1e-3)
            filter2 = (np.abs(df.T_b - TPoints[t]) < 1e-3)
            filter3 = (df.nbr_samples == nbr_samples)
            filter4 = (df.nbr_env == nbr_env)
            filter5 = (np.abs(df.confounder_strength - conf_strength) < 1e-3)

            tmp_df = df[filter1 & filter2 & filter3 & filter4 & filter5]
            assert len(tmp_df) == 1, 'There are more than one value for the reject rate'
            
            ZPoints[x][t] = tmp_df.reject_rate.values[0] # Assumes only one value per x,y combination


    # Print x,y and z values
    print(XPoints)
    print(TPoints)
    print(ZPoints)

    # Set x axis label for the contour plot
    plt.ylabel('Standard deviation $\sigma_{\\theta_U}$')

    # Set y axis label for the contour plot
    plt.xlabel('Standard deviation $\sigma_{\\theta_T}$')

    # Create contour lines or level curves using matplotlib.pyplot module
    cmap = plt.cm.get_cmap('YlGnBu')
    contours = plt.contourf(TPoints, XPoints, ZPoints, cmap=cmap, vmin=0, vmax=1)

    # Display z values on contour lines
    #plt.clabel(contours, inline=1, fontsize=10)
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap))
    cbar.set_label('Probability of detection')



def plot_curve(exp_res : dict, x_key : str, fix_val : int, label='', iter = None):

    rules = {
        'x_label' : {'nbr_env' : 'Number of environments', 'nbr_samples' : 'Number of samples per environment'},
        'fix_val' : {'nbr_env' : 'nbr_samples', 'nbr_samples' : 'nbr_env'}
    }

    for alg in exp_res:

        df = exp_res[alg]
        df = df[df[rules['fix_val'][x_key]] == fix_val ]

        p = df.reject_rate

        plt.plot(df[x_key], p, label=label, marker=marker_dict[alg])
        if iter:
            std = np.sqrt(p*(1-p)/iter)
            plt.fill_between(df[x_key], p-std, p+std, alpha=0.5)
   
    
    plt.xlabel(rules['x_label'][x_key])
    plt.ylabel('Probability of detection')
    plt.ylim([-.1,1.1])
    plt.legend()

