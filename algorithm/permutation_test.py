from cgi import test
from multiprocessing import Pool

import numpy as np
import pandas as pd

from pgmpy.estimators.CITests import g_sq

class PermutationConfounderTest:
    '''
    Based on Permutation Testing Improves Bayesian Network Learning (Tsamardinos and Bourboudakis)
    '''

    def __init__(self) -> None:
        pass 

    def test(data : dict, test = None, nbr_permutations_mcmc = 200, nbr_proc = 5,
                verbose = False) -> float:
        '''
        Goes through as many two-sample pairs as possible, and permutates contingency table to approximate the p-value.

        Uses G-test for cond. independence testing

        data : dict containing pd.Dataframe for each environment, the dataframe contains two variables: T, Y

        return: p-value
        '''

        test_var = ['Tn', 'Y1']
        cond_var = 'T1'

        nbr_samples = np.min([len(data[key]) for key in data])

        # PART 1: Collect two sample columns and compute G-statistic
        if verbose: print('Beginning PART 1 ...')
        samples = lambda key, var : data[key][var]
        
        
        two_sample_columns = []
        G_obs   = 0
        n_idx = 0
        while n_idx+1 < nbr_samples:
            
            # Get 2-sample column
            T1 = [samples(key,'T').iloc[n_idx] for key in data]
            Tn = [samples(key,'T').iloc[n_idx+1] for key in data]

            Y1 = [samples(key,'Y').iloc[n_idx] for key in data]
            Yn = [samples(key,'Y').iloc[n_idx+1] for key in data]

            
            df_tmp = pd.DataFrame.from_dict({'T1' : T1, 'Tn' : Tn, 'Y1' : Y1, 'Yn' : Yn})

            two_sample_columns.append(df_tmp)

            statistic, _, _ = g_sq(test_var[0], test_var[1], [cond_var], df_tmp, boolean=False)

            G_obs += statistic
            n_idx += 2

        if verbose: 
            print('PART 1 done')
            
        # PART 2

        G_permut_list = []

        with Pool(processes=nbr_proc) as pool:
            
            args_ = [[two_sample_columns, test_var, cond_var] for i in range(nbr_permutations_mcmc)]
            for G_permut_stat in pool.imap_unordered(PermutationConfounderTest.f, args_):
                G_permut_list.append(G_permut_stat)
            
        pval = 0
        for g_val in G_permut_list:

            if G_obs < g_val:
                pval += 1
            else:
                pass 
        
        pval = pval / len(G_permut_list)

        return pval


    def f(args : list):
        '''
        For parallelization
        '''
        two_sample_columns, test_var, cond_var = args[0], args[1], args[2]

        G_permut_tmp = 0

        for df_tmp in two_sample_columns:

            unique_cond_val = list(df_tmp[cond_var].unique())

            permuted_df_list = []

            # Randomly select which variable to permute
            if np.random.uniform(size=1) < 0.5:
                permut_var = test_var[0]
            else:
                permut_var = test_var[1]

            for v in unique_cond_val:
                
                cond_df_tmp = df_tmp[df_tmp[cond_var] == v].copy()
                
                cond_df_tmp.loc[:,permut_var] = np.random.permutation(cond_df_tmp[permut_var])

                permuted_df_list.append(cond_df_tmp)

            # Concatenate list of permutated dataframe
            permuted_df = pd.concat(permuted_df_list)

            statistic, _, _  = g_sq(test_var[0], test_var[1], [cond_var], permuted_df, boolean=False)

            G_permut_tmp += statistic

        return G_permut_tmp 
