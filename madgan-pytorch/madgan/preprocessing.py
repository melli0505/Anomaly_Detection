import csv

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def read_data(path: str):
    file = open(path, 'r')
    reader = csv.reader(file)
    data = []

    for row in reader:
        data.append(list(map(float, row)))

    return np.array(data, dtype=np.float32)

def reshape(window_size:int, data:np.array):
    if len(data.shape) > 1 and data.shape[1] == window_size:
        return data
    else:
        if len(data.shape) > 1 and data.shape[1] != window_size:
            data = data.reshape(data.shape[0] * data.shape[1])

        drop = data.shape[0] % window_size
        data_num = data.shape[0] // window_size

        return data[:-1 * drop].reshape(data_num, window_size)

def normalize(data:np.array):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler = scaler.fit(data)
    return scaler.transform(data)