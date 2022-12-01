import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer


def compute_G_statistic(X, Y):
    '''
    Compute G statistic for CI test X indep. of Y
    '''

    cont_table = create_contingency_table(X, Y)
    statistic, _, _, _ = chi2_contingency(cont_table, lambda_=0)
    return statistic

def create_contingency_table(X, Y):
    binsX, binsY = 2, 2
    # Adding +1 to each cell to avoid zero probabilities
    X = np.append(X, (0,0,1,1,1,1,1,1,1,1,1,1)) 
    Y = np.append(Y, (0,1,0,1,1,1,1,1,1,1,1,1))
    count = np.bincount(binsX * X + Y) 
    return count.reshape((binsX, binsY))

def get_propensity_score(X, T, label="", n_bins=10):
    '''
    Fit a LA regression model to predict T from X
    '''

    T = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit_transform(T.reshape(-1,1)).astype(int).ravel()
    model = LogisticRegression(max_iter=5000, multi_class='multinomial').fit(X, T)
    
    # Shift labels in T in case a bin has no samples in it
    corrected_T = T.copy()
    for i in range(n_bins):
        if i not in model.classes_:
            corrected_T[T>=i] -= 1     

    # compute propensity score
    prop_score = model.predict_log_proba(X)[:,1:]
    return pd.DataFrame(prop_score, columns=[f'propensity_score_{label}_{i}' for i in model.classes_[1:]])
