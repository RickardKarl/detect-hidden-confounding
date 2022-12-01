import abc
import numpy as np
import pandas as pd

class AbstractAlgorithm(abc.ABC):

    '''
    Abstract class for algorithm to detect hidden confounding
    '''

    @abc.abstractclassmethod
    def __init__(self) -> None:
        super().__init__()

    def test(self, data : dict, **param) -> float:
        pass

    def statistical_test(self, df : pd.DataFrame, X : str, Y : str, Z : str) -> float:
        pass