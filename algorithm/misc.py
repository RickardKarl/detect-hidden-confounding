from algorithm.general import PearsonConfounderTest
from algorithm.kernel.test import KernelConfounderTest

class PearsonPropensityConfounderTest(PearsonConfounderTest):
    
    def __init__(self) -> None:
        super().__init__(use_propensity_score=True)
    
class KernelPropensityConfounderTest(KernelConfounderTest):
        
    def __init__(self, epsilon=0.001, alpha=0.05) -> None:
        super().__init__(epsilon=epsilon, alpha=alpha, use_propensity_score=True)