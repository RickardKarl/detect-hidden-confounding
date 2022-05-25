
from random import random
from data.continuous import GaussianData
from tqdm import tqdm
from multiprocessing import Pool

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def save_results(df, file_name, timestamp):
    path = f'./results/{file_name}_{timestamp}.csv'
    df.to_csv(path)

'''
fast_experiment and fast2_experiment works very similarly, they just speed up different things depending on what I 
wanted to loop over

run does the actual experiment 

sample_efficiency used by vary_sample_env.ipynb

'''

def fast_experiment(dist_param_list : list,
               nbr_env : list,
               nbr_samples : list,
               conf_strength : list,
               SimulateClass,
               TestClass,
               repetitions = 20,
               sign_level = 0.05,
               nbr_proc = 4):

    df_list = []

    with Pool(processes=nbr_proc) as pool:

        
        args_ = [[d, nbr_env, nbr_samples, conf_strength, SimulateClass, TestClass, repetitions, sign_level] for d in dist_param_list]
        count = 0
        print(f'Progress: {count}/{len(dist_param_list)}', end='\r')
        for res in pool.imap_unordered(run, args_):
            print(f'Progress: {count}/{len(dist_param_list)}', end='\r')
            count   += 1
            df_list += res
        print(f'Progress: {count}/{len(dist_param_list)}', end='\r')
        experiment_results = pd.concat(df_list)

    return experiment_results

def fast2_experiment(dist_param: dict,
               nbr_env : list,
               nbr_samples : list,
               conf_strength : list,
               SimulateClass,
               TestClass,
               repetitions = 20,
               sign_level = 0.05,
               nbr_proc = 4,
               compute_bias = False):

    df_list = []
    
    args_ = [[dist_param, nbr_env, [n], conf_strength, 
                SimulateClass, TestClass, repetitions, sign_level, compute_bias] for n in nbr_samples]
    
    with Pool(processes=nbr_proc) as pool:
 
        count = 0
        print(f'Progress: {count}/{len(nbr_samples)}', end='\r')
        for res in pool.imap_unordered(run, args_):
            print(f'Progress: {count}/{len(nbr_samples)}', end='\r')
            count   += 1
            df_list += res
        print(f'Progress: {count}/{len(nbr_samples)}', end='\r')
        experiment_results = pd.concat(df_list)
    
    return experiment_results

def run(args):
    
    dist_param      = args[0]
    nbr_env         = args[1]
    nbr_samples     = args[2]
    conf_strength   = args[3]
    SimulateClass   = args[4]
    TestClass       = args[5]
    repetitions     = args[6]
    sign_level      = args[7]
    if len(args) == 9:
        compute_bias = args[8]
    else:
        compute_bias = False

    if SimulateClass == GaussianData:
        test = 'pearsonr'
    else:
        test = 'g_sq'

    df_list = []
    for e in nbr_env:
        for n in nbr_samples:
            for c_strength in conf_strength:
            
                rejection_rate = 0
                bias = []
                for iter in range(repetitions):
                    
                    # Sample data
                    data_gen = SimulateClass(e, dist_param)
                    data = data_gen.sample(n, c_strength)                
                    
                    # Run test 
                    pval = TestClass.test(data, test=test)
                    
                    # Check result
                    if pval < sign_level:
                        rejection_rate += 1
                    
                    if compute_bias:
                        tmp_bias = compute_confounding_bias(data)
                        bias.append(tmp_bias)

                # Compute average result                
                rejection_rate = rejection_rate / repetitions # Get average rejection rate
                
                # Save result across iterations
                tmp_df = pd.DataFrame({'nbr_env' : [e],
                                        'nbr_samples' : [n],
                                        'confounder_strength' : [c_strength],
                                        'reject_rate' : [rejection_rate],
                                        'desc' : data_gen.describe(),
                                        })

                if compute_bias:
                    tmp_df['avg_bias'] = np.mean(bias)
                    tmp_df['std_bias'] = np.std(bias)
                    
                for key in data_gen.dist_param:
                    # Either contains a float or a another dict
                    if type(data_gen.dist_param[key]) == dict:
                        for param in data_gen.dist_param[key]:
                            tmp_df[f'{key}_{param}'] = data_gen.dist_param[key][param]

                if 'X_alpha' in data_gen.dist_param:
                    tmp_df['X_alpha'] = data_gen.dist_param['X_alpha']
                if 'X_beta' in data_gen.dist_param:
                    tmp_df['X_beta'] = data_gen.dist_param['X_beta']
                if 'T_beta' in data_gen.dist_param:
                    tmp_df['T_beta'] = data_gen.dist_param['T_beta']
                if 'X_sigma' in data_gen.dist_param:
                    tmp_df['X_sigma'] = data_gen.dist_param['X_sigma']
                
                df_list.append(tmp_df)

    return df_list


def sample_efficiency(dist_param: dict,
                    nbr_env : list,
                    nbr_samples : list,
                    conf_strength : list,
                    SimulateClass,
                    algorithm_list : list,
                    repetitions = 20,
                    sign_level = 0.05):

    list_exp = {}

    for alg in tqdm(algorithm_list):
        
        alg_name = alg.__name__
        args = [dist_param, nbr_env, nbr_samples, conf_strength, SimulateClass, alg, repetitions, sign_level]
        res = run(args)
        
        list_exp[alg_name] = pd.concat(res)
    

    return list_exp


def compute_confounding_bias(data : dict):

    def samples(key, var): return data[key][var].values.reshape(-1,1)
    
    # Select first m_idx samples from each environment
    T = np.array([samples(key, 'T') for key in data]).flatten().reshape(-1,1)
    Y = np.array([samples(key, 'Y') for key in data]).flatten().reshape(-1,1)
    X = np.array([samples(key, 'X') for key in data]).flatten().reshape(-1,1)

    adjusted_ols = LinearRegression()
    unadjusted_ols = LinearRegression()
    

    adjusted_feat = np.stack([T, X], axis=1).squeeze()
        
    unadjusted_ols.fit(T, Y)
    adjusted_ols.fit(adjusted_feat, Y)
    unadjusted_ce = unadjusted_ols.coef_[0][0]
    adjusted_ce   = adjusted_ols.coef_[0][0]

    bias = unadjusted_ce - adjusted_ce

    return bias