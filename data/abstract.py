import abc
import numpy as np


class AbstractData(abc.ABC):
    
    '''
    Abstract class for simulating data
    '''

    @abc.abstractclassmethod
    def __init__(self, nbr_env : int, mechanisms : dict, seed : int) -> None:
        super().__init__()

    @abc.abstractclassmethod
    def sample(self, n_samples : int) -> np.ndarray:
        '''
        Sample simulated data
        '''
        pass


    def describe(self) -> str:
        '''
        Get a description of the object, including parameters, as a string
        '''
        pass
