import numpy as np

def normalize(signal: np.ndarray):
    # Normalize between 0 and 1
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))