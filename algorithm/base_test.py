import numpy as np
import pandas as pd
from scipy.stats import chi2

# Tests
from pgmpy.estimators.CITests import g_sq, pearsonr
from pingouin import partial_corr

class TwoSampleConfounderTest:

    allowed_test = ['g_sq', 'pearsonr']

    def __init__(self) -> None:
        pass

    def test(data : dict, test = 'g_sq', return_statistic = False) -> float:
        '''
        Uses one sample from each environment.

        Uses G-test for cond. independence testing

        data : dict containing pd.Dataframe for each environment, the dataframe contains two variables: T, Y

        return: p-value
        '''

        assert test in TwoSampleConfounderTest.allowed_test, f'{test} is not in the list of available tests'

        def samples(key, var): return data[key][var].values

        # Select first m_idx samples from each environment
        T1 = np.array([samples(key, 'T')[0] for key in data])
        Y1 = np.array([samples(key, 'Y')[0] for key in data])

        # Select the remaining samples in each environment
        Tn = np.array([samples(key, 'T')[1] for key in data])
        Yn = np.array([samples(key, 'Y')[1] for key in data])

        # Collect data in dict to save it as a dataframe (necessary for pg.partial_corr)
        tmp = {'Y1': Y1, 'T1': T1,
               'Yn': Yn, 'Tn': Tn}

        df = pd.DataFrame(data=tmp)


        if test == 'g_sq':
            stat, p_val, _ =  g_sq('Tn', 'Y1', ['T1'], df, boolean=False)
        elif test == 'pearsonr':
            #stat, p_val = pearsonr('Tn', 'Y1', ['T1'], df, boolean=False)
            res = partial_corr(df, x='Tn', y='Y1', covar='T1')
            stat = res['r'].values[0]
            p_val = res['p-val'].values[0]
        if return_statistic:
            return stat, p_val
        else:
            return p_val

class FullTwoSampleConfounderTest:

    allowed_test = ['g_sq']

    def __init__(self) -> None:
        pass

    def test(data : dict, test = 'g_sq', return_statistic = False) -> float:
        '''
        Adds upp all statistics from two-sample pairs in data.

        Uses G-test for cond. independence testing

        data : dict containing pd.Dataframe for each environment, the dataframe contains two variables: T, Y

        return: p-value
        '''

        assert test in FullTwoSampleConfounderTest.allowed_test, f'{test} is not in the list of available tests'

        nbr_samples = np.min([len(data[key]) for key in data]) # We use the number of samples that are the least in one of the env.
        samples = lambda key, var : data[key][var]
        
        # Collect two sample columns and compute G-statistic

        two_sample_columns = []
        total_G   = 0
        total_dof = 0
        n_idx     = 0

        while n_idx+1 < nbr_samples:
            
            # Get 2-sample column
            T1 = [samples(key,'T').iloc[n_idx] for key in data]
            Tn = [samples(key,'T').iloc[n_idx+1] for key in data]

            Y1 = [samples(key,'Y').iloc[n_idx] for key in data]
            Yn = [samples(key,'Y').iloc[n_idx+1] for key in data]

            
            df_tmp = pd.DataFrame.from_dict({'T1' : T1, 'Tn' : Tn, 'Y1' : Y1, 'Yn' : Yn})

            two_sample_columns.append(df_tmp)

            statistic, _, dof =  g_sq('Tn', 'Y1', ['T1'], df_tmp, boolean=False)

            total_G   += statistic
            total_dof += dof
            n_idx     += 2

        p_val = chi2.sf(total_G, total_dof)
        
        if return_statistic:
            return total_G, total_dof, p_val
        else:
            return p_val


