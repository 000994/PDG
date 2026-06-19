import numpy as np
import config as cfg

def z_score_normalize(X):
    mean = X.mean(axis=(1,2), keepdims=True)
    std = X.std(axis=(1,2), keepdims=True) + 1e-8
    return (X - mean) / std

def process_data(X_train, X_test):
    X_train = z_score_normalize(X_train)
    X_test = z_score_normalize(X_test)
    return X_train, X_test