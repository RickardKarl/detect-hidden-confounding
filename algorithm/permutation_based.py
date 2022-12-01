import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

from algorithm.general import ConfounderTest
from algorithm.utils import compute_G_statistic

class PermutationBasedTest(ConfounderTest):
    '''
    Based on Permutation Testing Improves Bayesian Network Learning (Tsamardinos and Bourboudakis)
    '''

    permutation_batch_size = 100

    def __init__(self, nbr_mc_batches = 5, nbr_proc = 5) -> None:
        super().__init__()

        self.nbr_mc_batches = nbr_mc_batches
        self.nbr_proc = nbr_proc

    def statistical_test(self, df: pd.DataFrame, X: str, Y: str, Z: list) -> dict:
        '''
        Implementation of Permutation-based G-test
        X, Y, Z are the labels of the dataframe where to test X indep. of Y given Z
        '''

        assert len(Z) == 1, 'This implementation is not compatible with having observed confounders at the moment'
        Z = Z[0]
        
        data = df.values.astype(np.int16)
        
        # Get indices for variables matching to matrix
        test_var0_idx = df.columns.get_loc(X)
        test_var1_idx = df.columns.get_loc(Y)
        cond_var_idx = df.columns.get_loc(Z)

        # Compute G test statistic for observed data
        G_obs = 0
        # Loop over conditioned variable values
        for val in np.unique(data[:,cond_var_idx]):
            
            mask = (data[:,cond_var_idx] == val).squeeze() # condition on val
            G_obs += compute_G_statistic(data[mask, test_var0_idx], data[mask, test_var1_idx])

        G_permut_list = self.permute_parallelization(data, test_var0_idx, test_var1_idx, cond_var_idx)

        # Compute pval
        counts = 0
        for g_val in G_permut_list:

            if G_obs < g_val:
                counts += 1
            else:
                pass 
    
        pval = counts / len(G_permut_list)
        
        return {'pval' : pval, 'G_obs' : G_obs, 'G_permut' : G_permut_list}

    def permute_parallelization(self, data_matrix, test_var0_idx, test_var1_idx, cond_var_idx):

        '''
        TODO write desc
        '''

        G_permut_list = []
        with Pool(processes=self.nbr_proc) as pool:
            
            args_ = [[data_matrix.copy(), [test_var0_idx, test_var1_idx], cond_var_idx] for _ in range(self.nbr_mc_batches)]
            for G_permut_stat in pool.imap_unordered(PermutationBasedTest.permutate, args_):
                G_permut_list.append(G_permut_stat)

        # Merge the lists inside G_permut_list
        G_permut_list = list((itertools.chain.from_iterable(G_permut_list)))

        return G_permut_list


    def permutate(args : list):
        '''
        TODO: Write desc
        '''

        data, test_var_idx, cond_var_idx = args[0], args[1], args[2]

        # Get unique values
        unique_val = np.unique(data)

        G_permut_list = []
        for _ in range(PermutationBasedTest.permutation_batch_size):

            # Randomly choose which variable to permute over

            unif_var = np.random.uniform()
            if unif_var < 0.5:
                permut_var_idx = test_var_idx[0]
            else:
                permut_var_idx = test_var_idx[1]

            # Variable for saving test statistic 
            G_permut_tmp = 0

            # Loop over conditioned variable values
            for val in unique_val:
                
                mask = (data[:,cond_var_idx] == val)
                # Permute table
                data[mask, permut_var_idx] = np.random.permutation(data[mask, permut_var_idx])
                
                # Compute G-test statistics
                G_permut_tmp += compute_G_statistic(data[mask, test_var_idx[0]], data[mask, test_var_idx[1]])
        
            G_permut_list.append(G_permut_tmp)

        return G_permut_list 