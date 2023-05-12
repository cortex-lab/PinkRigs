import numpy as np


def get_VE(actual,predicted):
    """
    calculating variance explained for any sort of array. 
    """
    actual_=np.ravel(actual)
    predicted_ = np.ravel(predicted)
    VE = 1-(np.var(actual_-predicted_)/np.var(actual_))
    return VE
