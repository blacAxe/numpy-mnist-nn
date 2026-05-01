import pandas as pd
import numpy as np

def load_data():
    # Read CSV file 
    data = pd.read_csv('train.csv')
    data = np.array(data)

    m, n = data.shape

    # Shuffle rows so training is not biased by original order
    np.random.shuffle(data)

    # Use first 1000 samples as validation
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]              # labels
    X_dev = data_dev[1:n] / 255.     # normalize pixel values

    # Remaining data used for training
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.

    return X_train, Y_train, X_dev, Y_dev