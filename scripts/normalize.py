import numpy as np
import pandas as pd
from manual import mean, variance, std_dev

def Z_Score_normalize(arr):
    m = mean(arr)
    s = std_dev(arr)
    return (arr - m) / s

def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)