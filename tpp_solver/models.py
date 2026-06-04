"""Curve-fitting model functions."""
import numpy as np

def sigmoid(T, a, b, plateau):
    return a / (1 + np.exp(-(T - b))) + plateau

def paper_sigmoidal(T, A1, A2, Tm):
    return A2 + (A1 - A2) / (1 + np.exp((T - Tm)))
