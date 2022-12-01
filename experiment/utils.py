
from email import header
import os
from multiprocessing import Pool
from random import random
from statistics import mode

import numpy as np
import pandas as pd
from data.linear.continuous import GaussianLinearData
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


def save_results(df, path, append=False):

    mode = 'a' if append else 'w'
    header = (not os.path.exists(path)) or (not append)
    df.to_csv(path, mode=mode, header=header) # only use header if we write to file first time


def run(args, save_during_run=None):

    dist_param = args[0]    # dict
    nbr_env = args[1]       # list
    nbr_samples = args[2]   # list
    conf_strength = args[3] # list
    SimulateClass = args[4] # class
    test_method = args[5]   # class instance
    repetitions = args[6]   # int
    sign_level = args[7]    # float
    if len(args) >= 9: 
        compute_bias = args[8] # bool
    else:
        compute_bias = False
    if len(args) >= 10:
        vary_both_mechanisms = args[9] # bool
    else:
        vary_both_mechanisms = False

    df_list = []
    for e in nbr_env:
        for n in nbr_samples:
            for c_strength in conf_strength:

                rejection_rate = 0
                bias = []
                for _ in range(repetitions):

                    # Sample data
                    data_gen = SimulateClass(e, dist_param)
                    if vary_both_mechanisms:
                        data = data_gen.sample(n, Y_conf=c_strength, T_conf=c_strength)
                    else:
                        data = data_gen.sample(n, Y_conf=c_strength)

                    # Run test
                    res = test_method.test(data)
                    pval = res['pval']

                    # Check result
                    if pval < sign_level:
                        rejection_rate += 1

                    if compute_bias:
                        tmp_bias = compute_confounding_bias(data)
                        bias.append(tmp_bias)

                # Compute average result
                rejection_rate = rejection_rate / repetitions  # Get average rejection rate

                # Save result across iterations
                tmp_df = pd.DataFrame({'nbr_env': [e],
                                       'nbr_samples': [n],
                                       'confounder_strength': [c_strength],
                                       'reject_rate': [rejection_rate],
                                       'desc': data_gen.describe(),
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

                if save_during_run:
                    save_results(tmp_df, path=save_during_run, append=True)



    return df_list


def compute_confounding_bias(data: dict):

    def samples(key, var): return data[key][var].values.reshape(-1, 1)

    # Select first m_idx samples from each environment
    T = np.array([samples(key, 'T') for key in data]).flatten().reshape(-1, 1)
    Y = np.array([samples(key, 'Y') for key in data]).flatten().reshape(-1, 1)
    X = np.array([samples(key, 'X') for key in data]).flatten().reshape(-1, 1)

    adjusted_ols = LinearRegression()
    unadjusted_ols = LinearRegression()

    adjusted_feat = np.stack([T, X], axis=1).squeeze()

    unadjusted_ols.fit(T, Y)
    adjusted_ols.fit(adjusted_feat, Y)
    unadjusted_ce = unadjusted_ols.coef_[0][0]
    adjusted_ce = adjusted_ols.coef_[0][0]

    bias = unadjusted_ce - adjusted_ce

    return bias
