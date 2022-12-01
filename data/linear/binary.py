from re import A
from data.abstract import AbstractData

import numpy as np
import pandas as pd


class BinaryLinearData(AbstractData):

    required_dist_param = ['a', 'b']

    def __init__(self, nbr_env: int, dist_param: dict, seed=None) -> None:
        '''
        dist_param : 'a' : float, 'b' : float
        '''

        self.nbr_env = nbr_env

        # Check dist param
        self.dist_param = dist_param

        for m in ['T', 'X', 'Y']:
            for p in BinaryLinearData.required_dist_param:
                assert p in self.dist_param[m], f'Missing key {p} in dist_param[{m}]'

        # Set seed
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Generate parameters for each environment
        self.params = self.generate_params()

    def generate_params(self) -> dict:

        var_mech = {}

        # Noise distribution
        for m in ['T', 'X', 'Y']:
            a, b = self.dist_param[m]['a'], self.dist_param[m]['b']
            var_mech[f'{m}_mu'] = self.rng.normal(a, b, size=(self.nbr_env, 1))  # For distribution of mechanism


        ##  Deterministic values (TODO) 
        var_mech['X_alpha'] = self.dist_param['X_alpha'] if 'X_alpha' in self.dist_param else 1  # effect of X on T
        var_mech['X_beta']  = self.dist_param['X_beta']  if 'X_beta'  in self.dist_param else 1  # effect of X on Y
        var_mech['T_beta']  = self.dist_param['T_beta']  if 'T_beta'  in self.dist_param else 1  # effect of T on Y
        var_mech['X_sigma'] = self.dist_param['X_sigma'] if 'X_sigma' in self.dist_param else 1  # noise variance for X 

        return var_mech

    def sample(self, nbr_samples: int, Y_conf: float, T_conf = 1.0) -> dict:
        '''
        T_conf (float) : decides strength of confounding to T, zero means no confounding
        Y_conf (float) : decides strength of confounding to Y, zero means no confounding
        '''

        data = {}

        def sigmoid(v): return 1/(1+np.exp(-v))

        for i in range(self.nbr_env):

            # Retrieve parameters
            x_mu = self.params['X_mu'][i]
            t_mu = self.params['T_mu'][i]
            y_mu = self.params['Y_mu'][i]
            
            # Currently deterministic
            x_alpha = self.params['X_alpha']
            x_beta  = self.params['X_beta']
            t_beta  = self.params['T_beta']
            x_sigma = self.params['X_sigma']

            # Sample data
            e_X = self.rng.normal(x_mu, x_sigma, size=(nbr_samples))
            Xe = e_X 

            prob_Te = sigmoid(T_conf * x_alpha * Xe + t_mu)
            Te = self.rng.binomial(1, prob_Te) 

            prob_Ye =     sigmoid(Y_conf * x_beta * Xe + t_beta * Te + y_mu)
            Ye = self.rng.binomial(1, prob_Ye)

            # Save data
            tmp = {'X': Xe, 'Y': Ye, 'T': Te,
                    'x_mu' : x_mu*np.ones(Xe.shape), 't_mu' : t_mu*np.ones(Xe.shape), 'y_mu' : y_mu*np.ones(Xe.shape),
                    'x_sigma' : x_sigma*np.ones(Xe.shape)}

            data[i] = pd.DataFrame(tmp)

        return data

