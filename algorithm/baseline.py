import itertools

import numpy as np
import pandas as pd
from pgmpy.estimators.CITests import g_sq, pearsonr

from algorithm.abstract import AbstractAlgorithm


class EnvironmentTest(AbstractAlgorithm):

    allowed_test = ['g_sq', 'pearsonr']

    def __init__(self) -> None:
        pass

    def test(self, data : dict, observed_covariates = [], test = 'g_sq') -> float:
        '''
        Check whether environment is independent of outcome given treatment

        Uses G-test for cond. independence testing

        data : dict containing pd.Dataframe for each environment, the dataframe contains two variables: T, Y

        return: p-value
        '''

        assert test in EnvironmentTest.allowed_test, f'{test} is not in the list of available tests'

        def samples(key, var): return list(data[key][var].values)


        T = [samples(key, 'T') for key in data]
        Y = [samples(key, 'Y') for key in data]
        E = [[i for _ in range(len(samples(key, 'T')))] for i, key in enumerate(data)]
        
        # Unpack lists
        T = itertools.chain(*T)
        Y = itertools.chain(*Y)
        E = itertools.chain(*E)
        

        # Collect data in dict to save it as a dataframe (necessary for pg.partial_corr)
        tmp = {'T': T, 'Y': Y, 'E': E}

        # Add additional observed covariates
        for var_name in observed_covariates:
            X = [samples(key, var_name) for key in data]
            tmp[var_name] = itertools.chain(*X)

        df = pd.DataFrame(data=tmp)

        cond_var = observed_covariates + ['T']
        pval = self.statistical_test(df, 'Y', 'E', cond_var)

        return {'pval' : pval}

    def statistical_test(self, df: pd.DataFrame, X: str, Y: str, Z: list) -> float:
    
        raise NotImplementedError('Use a class which inherits EnvironmentTest class with an implementation of this method')


class GEnvironmentTest(EnvironmentTest):

    def __init__(self) -> None:
        super().__init__()

    def statistical_test(self, df: pd.DataFrame, X: str, Y: str, Z: list) -> float:

        _, pval, _ =  g_sq(X, Y, Z, df, boolean=False)
        return pval

class PearsonEnvironmentTest(EnvironmentTest):

    def __init__(self) -> None:
        super().__init__()

    def statistical_test(self, df: pd.DataFrame, X: str, Y: str, Z: list) -> float:

        _, pval =  pearsonr(X, Y, Z, df, boolean=False)
        return pval
