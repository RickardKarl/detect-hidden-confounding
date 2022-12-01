
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF

from algorithm.general import ConfounderTest
from algorithm.kernel.kcit import perform_kcit, ImplementedKCITSchemes
from algorithm.kernel.kernel_wrapper import KernelWrapper

class KernelConfounderTest(ConfounderTest):

    def __init__(self, epsilon=0.001, alpha=0.05, use_propensity_score=False) -> None:
        super().__init__(use_propensity_score=use_propensity_score)
        self.epsilon = epsilon
        self.alpha   = alpha

    def statistical_test(self, df: pd.DataFrame, X: str, Y: str, Z: list) -> float:
        
        data_x = df[X].values.reshape(-1,1)
        data_y = df[Y].values.reshape(-1,1) 
        data_z = df[Z].values

        length_scale_kx = np.median(np.abs(pdist(data_x)))
        kernel_kx = KernelWrapper(RBF(length_scale=length_scale_kx))
        length_scale_ky = np.median(np.abs(pdist(data_y)))
        kernel_ky = KernelWrapper(RBF(length_scale=length_scale_ky))
        length_scale_kz = np.median(np.abs(pdist(data_z)))
        kernel_kz = KernelWrapper(RBF(length_scale=length_scale_kz))
    
        
        res = perform_kcit(data_x, data_y, data_z,
                            kernel_kx, kernel_ky, kernel_kz,
                            self.epsilon, 
                            self.alpha,
                            ImplementedKCITSchemes.GAMMA
                            )

        pval = res['pvalue']
        #print(pval)
        return {'pval' : pval}


