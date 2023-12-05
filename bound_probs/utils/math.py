import numpy as np

def softplus_np(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def inverse_softplus_np(x):
    return np.log(np.exp(x) - 1.)