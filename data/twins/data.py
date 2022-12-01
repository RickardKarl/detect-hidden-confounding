import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from data.abstract import AbstractData


def load_data(data_folder) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # processed data by Louizos et al (2017)
    # drop unnamed column, it is same as index
    df_treatment = pd.read_csv(os.path.join(
        data_folder, 'twin_pairs_T_3years_samesex.csv')).drop('Unnamed: 0', axis=1)
    df_outcome = pd.read_csv(os.path.join(
        data_folder, 'twin_pairs_Y_3years_samesex.csv')).drop('Unnamed: 0', axis=1)
    df_covar = pd.read_csv(os.path.join(data_folder, 'twin_pairs_X_3years_samesex.csv')).drop(
        ['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

    # drop id of infants in covariates
    df_covar.drop(['infant_id_0', 'infant_id_1', 'data_year'],
                  axis=1, inplace=True)

    # Impute missing valuess
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed = imputer.fit_transform(df_covar)
    df_covar = pd.DataFrame(imputed, columns=df_covar.columns)

    return df_treatment, df_outcome, df_covar


class TwinsData(AbstractData):

    COVAR_LIST = ['birmon', 'brstate',
                  'dfageq', 'dlivord_min',
                  'dtotord_min', 'gestat10', 'mager8',
                  'meduc6', 'mplbir', 'nprevistq', 'stoccfipb']
    
    # Removed data_year since it causes issues with having environment brstate
    # Removed brstate_reg since it is almost (?) a duplicate of brstate

    def __init__(self, data_folder: str, env_label='brstate', nbr_confounders=None, seed=None) -> None:
        '''
        env_label (str) : name of variable for which to define different environments (should be categorical)
                          potential environments: birmon (birth month), brstate (birth state), stoccfipb, TODO (data year and month)
        '''

        # Read Twins data
        self.df_treatment, self.df_outcome, self.df_covar = load_data(data_folder)  # deterministic output
        self.env_label = env_label
        self.env_idx = self.get_environments(env_label=env_label)

        # Set seed
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.nbr_confounders = nbr_confounders
        self.randomly_select_confounders()

        # Dict for distribution parameters
        self.params = {}

    def get_environments(self, env_label: str) -> dict:

        # Find labels
        uniq_env = self.df_covar[env_label].unique()
        env_idx = {}
        threshold_nbr_samples = 100

        for e in uniq_env:
            sample_idx = self.df_covar[self.df_covar[env_label]== e].index.values
            if len(sample_idx) >= threshold_nbr_samples:
                env_idx[e] = sample_idx

        return env_idx

    def randomly_select_confounders(self):

        covar_list = TwinsData.COVAR_LIST
        if self.env_label in covar_list:
            covar_list.remove(self.env_label)

        if self.nbr_confounders is not None:
            assert self.nbr_confounders <= len(covar_list), 'nbr_confounders is larger than available number of confounders'
            self.covar_list = list(self.rng.choice(covar_list, self.nbr_confounders, replace=False))
        else:
            self.covar_list = covar_list


    def generate_params(self, nbr_changes=None, reset=True) -> dict:
        '''
        nbr_changes (int): number of parameter that we change at the same time to control sparsity (or lack thereof)
        '''

        if reset:
            var_mech = {}
        else:
            var_mech = self.params

        mechanism_list = ['g_T']
        for var in self.covar_list:
            mechanism_list.append(f'fT_{var}')
            mechanism_list.append(f'fY_{var}')

        if nbr_changes is None or reset:
            changing_mechanism = mechanism_list
        else:
            if nbr_changes > len(mechanism_list):
                print('Warning: nbr_changes > len(mechanism_list)')
                nbr_changes = len(mechanism_list)
            changing_mechanism = self.rng.choice(mechanism_list, nbr_changes, replace=False)

        possible_functions = [lambda x: np.tanh(x), lambda x : np.cos(x), lambda x: x, lambda x: x**2]

        # T ~ f(X) + noise
        for var in self.covar_list:
            if f'fT_{var}' in changing_mechanism:
                W = self.rng.uniform(1, 5, size=(1, 1))
                var_mech[f'fT_{var}'] = lambda x: W * self.rng.choice(possible_functions)(x)

        # Y ~ f(X) + g(T) + noise
        for var in self.covar_list:
            if f'fY_{var}' in changing_mechanism:
                MY = self.rng.uniform(1, 5, size=(1, 1))
                var_mech[f'fY_{var}'] = lambda x: MY * self.rng.choice(possible_functions)(x)

        if 'g_T' in changing_mechanism:
            MT = self.rng.uniform(1, 2, size=(1, 1))
            var_mech['g_T'] = lambda x: MT*x

        return var_mech

    def sample(self, conf_strength: float, resample_mechanisms=False, binary=False, nbr_changes=None) -> dict:
        '''
        nbr_changes (int): number of parameter that we change at the same time to control sparsity (or lack thereof)
        '''

        def sigmoid(x): return 1/(1+np.exp(-x))
        observational_data = {}
        true_ate = {}

        # Generate parameters for first time
        self.params = self.generate_params()

        for e in self.env_idx:

            if resample_mechanisms:  # re-generate parameters for each environment
                env_params = self.generate_params(nbr_changes=nbr_changes, reset=False)
            else:
                env_params = self.params
            
            sample_idx = self.env_idx[e]
            nbr_samples = len(sample_idx)
            nbr_covar = len(self.covar_list)

            # Load covariates
            covar = np.zeros((nbr_samples, nbr_covar))
            for i, c in enumerate(self.covar_list):
                covar[:, i] = self.df_covar[c].values[sample_idx]
                if len(np.unique(covar[:, i])) == 1:
                    covar[:, i] = 5*covar[:,i] - np.mean(covar[:, i])
                else:
                    covar[:, i] = 5*(covar[:, i] - np.mean(covar[:, i])) / (np.max(covar[:, i]) - np.min(covar[:, i]))

            T = np.zeros((nbr_samples, 1))
            for i, c in enumerate(self.covar_list):
                T += env_params[f'fT_{c}'](covar[:, i]).reshape(-1, 1)

            if not binary:
                T += self.rng.normal(0, 1/4, size=(nbr_samples, 1))
            else:
                T = sigmoid(T)
                T = np.clip(T, 0.05, 0.95)
                T = self.rng.binomial(1, T)

            Y = env_params['g_T'](T).reshape(-1, 1)
            for i, c in enumerate(self.covar_list):
                Y += conf_strength * \
                    env_params[f'fY_{c}'](covar[:, i]).reshape(-1, 1)

            if not binary:
                Y += self.rng.normal(0, 1/4, size=(nbr_samples, 1))
            else:
                Y = sigmoid(Y)
                Y = np.clip(Y, 0.05, 0.95)
                Y = self.rng.binomial(1, Y)

            T = T.squeeze()
            Y = Y.squeeze()

            # Save T and Y
            observational_data[int(e)] = pd.DataFrame({'T': T, 'Y': Y})

            # Save covariates
            for i, c in enumerate(self.covar_list):
                observational_data[int(e)][c] = covar[:, i].squeeze()

            true_ate[int(e)] = (env_params['g_T'](1) - env_params['g_T'](0))

        return {'observational_data': observational_data, 'true_ate': true_ate}


if __name__ == '__main__':
    pass