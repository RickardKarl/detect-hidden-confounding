from xmlrpc.client import boolean
import numpy as np
import pandas as pd

from pgmpy.estimators.CITests import g_sq, pearsonr

class EnvironmentTest:

    allowed_test = ['g_sq', 'pearsonr']

    def __init__(self) -> None:
        pass

    def test(data : dict, test = 'g_sq', return_statistic = False) -> float:
        '''
        Check whether environment is independent of outcome given treatment

        Uses G-test for cond. independence testing

        data : dict containing pd.Dataframe for each environment, the dataframe contains two variables: T, Y

        return: p-value
        '''

        assert test in EnvironmentTest.allowed_test, f'{test} is not in the list of available tests'

        def samples(key, var): return data[key][var].values


        T = np.array([samples(key, 'T') for key in data]).flatten()
        Y = np.array([samples(key, 'Y') for key in data]).flatten()
        E = np.array([i*np.ones(samples(key, 'T').shape, dtype=np.int8) for i, key in enumerate(data)]).flatten()


        # Collect data in dict to save it as a dataframe (necessary for pg.partial_corr)
        tmp = {'T': T, 'Y': Y, 'E': E}

        df = pd.DataFrame(data=tmp)

        if test == 'g_sq':
            stat, p_val, dof =  g_sq('Y', 'E', ['T'], df, boolean=False)
        else:
            stat, p_val = pearsonr('Y', 'E', ['T'], df, boolean=False)
            dof = None
        
        if return_statistic:
            return stat, p_val, dof
        else:
            return p_val