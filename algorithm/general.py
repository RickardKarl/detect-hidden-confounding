from re import I
import numpy as np
import pandas as pd

from scipy.stats import chi2

from pgmpy.estimators.CITests import g_sq, pearsonr
from statsmodels.stats.contingency_tables import StratifiedTable

from algorithm.abstract import AbstractAlgorithm
from algorithm.utils import create_contingency_table, get_propensity_score


class ConfounderTest(AbstractAlgorithm):

    def __init__(self, use_propensity_score=False) -> None:
        self.use_propensity_score = use_propensity_score

    def test(self, data: dict, observed_covariates=[], max_tests = None, min_env_size = 2, alpha=0.05) -> float:

        # Match samples columns into pair with about the same sample size

        # Sort environments w.r.t to sample size
        sample_sizes = [(key, len(data[key])) for key in data]
        sample_sizes.sort(key=lambda y: y[1], reverse=True)
        max_nbr_samples = sample_sizes[1][1]
        def get_samples(key, var): return data[key][var].values

        if max_tests is None: 
            max_tests = max_nbr_samples # ensures we use all samples to test   
        
        df_list = []
        n = 0
        while n < max_nbr_samples-1:

            tmp_key_list = [key for key, ns in sample_sizes if ns > n+1]

            if len(tmp_key_list) < min_env_size:
                n += 2
                continue

            # Select first sample in each environment
            T_i = np.array([get_samples(key, 'T')[n] for key in tmp_key_list])
            Y_i = np.array([get_samples(key, 'Y')[n] for key in tmp_key_list])

            # Select second sample in each environment
            T_j = np.array([get_samples(key, 'T')[n+1]
                           for key in tmp_key_list])
            Y_j = np.array([get_samples(key, 'Y')[n+1]
                           for key in tmp_key_list])

            data_dict = {'Y_i': Y_i, 'T_i': T_i, 'T_j': T_j, 'Y_j': Y_j}

            # Select observed confounders
            for var_name in observed_covariates:
                X_i = np.array([get_samples(key, var_name)[n]
                               for key in tmp_key_list])
                X_j = np.array([get_samples(key, var_name)[n+1]
                               for key in tmp_key_list])
                data_dict[f'{var_name}_i'] = X_i
                data_dict[f'{var_name}_j'] = X_j

            # Collect data in dict to save it as a dataframe
            df_list.append(pd.DataFrame(data=data_dict))

            n += 2
            
            if len(df_list) >= max_tests:
                break
            
        # Add covariates to the list of conditional variables
        cond_var = []
        for var_name in observed_covariates:
            cond_var.append(f'{var_name}_i')
            cond_var.append(f'{var_name}_j')

        # Use Fisher's method to combine multiple tests for the same null hypothesis

        pval_list = []
        debug = []
        for df_tmp in df_list:

            if self.use_propensity_score and len(observed_covariates) > 0:

                cond_var_i = [var for var in cond_var if var.endswith('_i')]
                cond_var_j = [var for var in cond_var if var.endswith('_j')]
                df_i = get_propensity_score(df_tmp[cond_var_i].values, df_tmp['T_i'].values, label='i')
                df_j = get_propensity_score(df_tmp[cond_var_j].values, df_tmp['T_j'].values, label='j')
                df_tmp = pd.concat([df_tmp, df_i, df_j], axis=1) 
                res = self.statistical_test(df_tmp, 'T_j', 'Y_i', ['T_i'] + list(df_i.columns) + list(df_j.columns))

            else:
                res = self.statistical_test(df_tmp, 'T_j', 'Y_i', cond_var + ['T_i'])

            pval_list.append(res['pval'])

            debug.append((res['pval'], df_tmp))

            if np.isnan(res['pval']):
                print(res['pval'])
            

        # scaled log-sum of p-values has chi-squared distribution under null
        X = np.sum([-2*np.log(pval + 1e-3) for pval in pval_list])
        threshold = chi2.ppf(1-alpha, df=2*len(pval_list))
        pval = 1 - chi2.cdf(X, df=2*len(pval_list))
        

        return {'pval': pval, 'pval_list': pval_list, 'X': X, 'threshold': threshold, 'debug': debug}

    def statistical_test(self, df: pd.DataFrame, X: str, Y: str, Z: list) -> float:

        raise NotImplementedError(
            'Use a class which inherits ConfounderTest class with an implementation of this method')


class GConfounderTest(ConfounderTest):

    def __init__(self) -> None:
        super().__init__(use_propensity_score=True)

    def statistical_test(self, df: pd.DataFrame, X: str, Y: str, Z: list) -> dict:
        '''
        Implementation of G-test
        X, Y, Z are the labels of the dataframe where to test X indep. of Y given Z
        '''

        _, pval, _ = g_sq(X, Y, Z, df, boolean=False)

        return {'pval': pval}


class PearsonConfounderTest(ConfounderTest):

    def __init__(self, use_propensity_score=False) -> None:
        super().__init__(use_propensity_score=use_propensity_score)

    def statistical_test(self, df: pd.DataFrame, X: str, Y: str, Z: list) -> dict:
        '''
        Implementation of Pearson's correlation coef. test
        X, Y, Z are the labels of the dataframe where to test X indep. of Y given Z
        '''

        _, pval =  pearsonr(X, Y, Z, df, boolean=False)
        return {'pval': pval}
