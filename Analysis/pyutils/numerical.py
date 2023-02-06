
import numpy as np

def ismember(a, b):
    """
    equivalent of np.isin but returns indices as in the matlab ismember function
    returns an array containing logical 1 (true) where the data in A is B
    also returns the location of members in b such as a[lia] == b[locb]
    :param a: 1d - array
    :param b: 1d - array
    :return: isin, locb
    """
    lia = np.isin(a, b)
    aun, _, iuainv = np.unique(a[lia], return_index=True, return_inverse=True)
    _, ibu, iau = np.intersect1d(b, aun, return_indices=True)
    locb = ibu[iuainv]
    return lia, locb
