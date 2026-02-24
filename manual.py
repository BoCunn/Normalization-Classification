import numpy as np

def mean(arr):
    return np.sum(arr) / len(arr)

def variance(arr):
    m = mean(arr)
    return np.sum((arr - m) ** 2) / len(arr)

def std_dev(arr):
    return np.sqrt(variance(arr))

