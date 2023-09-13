import numpy as np
import os
import argparse
from pathlib import Path

if __name__ == "__main__":
    # Prepare synthetic data
    seed = 7
    np.random.seed(seed)
    numpy_dtype = np.float32
    num_samples = 20000
    input_size = 64
    data_dir = "data"

    X = np.random.randn(num_samples, input_size).astype(numpy_dtype)
    w = np.random.randn(input_size).astype(numpy_dtype)
    y = np.matmul(X, w[:, np.newaxis])[:,0] + np.random.randn(num_samples)*0.001
    y = y.astype(numpy_dtype)
    train_percent = 0.9
    X_train = X[:int(train_percent*num_samples),:]
    y_train = y[:int(train_percent*num_samples)]
    X_valid = X[int(train_percent*num_samples):,:]
    y_valid = y[int(train_percent*num_samples):]
    
    Path("./data").mkdir(parents=False, exist_ok=True)
    np.save(os.path.join(data_dir, "xtrain.npy"), X_train)
    np.save(os.path.join(data_dir, "ytrain.npy"), y_train)
    np.save(os.path.join(data_dir, "xvalid.npy"), X_valid)
    np.save(os.path.join(data_dir, "yvalid.npy"), y_valid)



