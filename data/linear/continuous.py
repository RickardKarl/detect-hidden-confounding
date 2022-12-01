from data.linear.binary import BinaryLinearData

import numpy as np
import pandas as pd


class GaussianLinearData(BinaryLinearData):

    def __init__(self, nbr_env: int, dist_param: dict, seed=None) -> None:
        super().__init__(nbr_env, dist_param, seed)

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
        var_mech['X_sigma'] = self.dist_param['X_sigma'] if 'X_sigma' in self.dist_param else 1 # noise variance for X (intentionally 2) 
        var_mech['Y_sigma'] = self.dist_param['Y_sigma'] if 'Y_sigma' in self.dist_param else 1  # noise variance for Y 
        var_mech['T_sigma'] = self.dist_param['T_sigma'] if 'T_sigma' in self.dist_param else 1  # noise variance for T 

        return var_mech

    def sample(self, nbr_samples: int, Y_conf: float, T_conf = 1.0) -> dict:
        '''
        conf_strength (float) : decides strength of confounder, zero means no confounding
        '''

        data = {}

        # Currently deterministic
        x_alpha = self.params['X_alpha']
        x_beta  = self.params['X_beta']
        t_beta  = self.params['T_beta']
        x_sigma = self.params['X_sigma']
        y_sigma = self.params['Y_sigma']
        t_sigma = self.params['T_sigma']
        
        for i in range(self.nbr_env):

            # Retrieve parameters
            x_mu = self.params['X_mu'][i]
            t_mu = self.params['T_mu'][i]
            y_mu = self.params['Y_mu'][i]
        

            # Sample data
            Xe = self.rng.normal(x_mu, x_sigma, size=(nbr_samples)) 

            e_T = self.rng.normal(t_mu, t_sigma, size=(nbr_samples))                
            Te = T_conf * x_alpha * Xe + e_T  

            e_Y = self.rng.normal(y_mu, y_sigma, size=(nbr_samples))
            Ye = Y_conf * x_beta * Xe + t_beta * Te + e_Y

            # Save data
            tmp = {'X': Xe, 'Y': Ye, 'T': Te,
                    'x_mu' : x_mu*np.ones(Xe.shape), 't_mu' : t_mu*np.ones(Xe.shape), 'y_mu' : y_mu*np.ones(Xe.shape),
                    'x_sigma' : x_sigma*np.ones(Xe.shape), 't_sigma' : t_sigma*np.ones(Xe.shape), 'y_sigma' : y_sigma*np.ones(Xe.shape)}

            data[i] = pd.DataFrame(tmp)

        return data
