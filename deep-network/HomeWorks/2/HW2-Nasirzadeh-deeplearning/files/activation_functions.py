
import numpy as np
import matplotlib.pyplot as plt

def hardlims(n):
    """Symmetrical Hard Limit: -1 for n<0, +1 for n≥0"""
    return np.where(n < 0, -1, 1)

def pureline(n):
    """Linear: a = n"""
    return n

def satlins(n):
    """Saturating Linear: 0 for n<0, n for 0≤n≤1, 1 for n>1"""
    return np.where(n < 0, 0, np.where(n > 1, 1, n))

def poslin(n):
    """Positive Linear: 0 for n<0, n for n≥0"""
    return np.where(n < 0, 0, n)