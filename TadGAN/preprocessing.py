import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

def make_abnormal_label(data):
    """
    make abnormal label as 0
    * this function is only for no fault data

    Args:
        data (list): signal dataset
    """
    data['abnormal'] = 0
    data.columns = ['', 'signal', 'abnormal']
    data = data.drop([''], axis=1)
    print(data.head)
    return data

def noise_control(data):
    array_data = np.array(data['signal'], dtype=np.float64)
    print(array_data.shape)
    # array_data = np.float32(array_data[1:])
    N = 10
    Wn = 0.4
    B, A = butter(N, Wn, output='ba')
    signal = filtfilt(B, A, array_data)
    print(signal.shape)

    data['signal'] = signal
    return data


def preprocessing(dataset):
    data = make_abnormal_label(dataset)
    data = noise_control(data)

    print(data.head)
    plt.plot(dataset['signal'])
    plt.plot(data['signal'])
    plt.show()


dataset = pd.read_csv('C:/Users/dk866/Desktop/bearing_test/data/set2_b1_outer_race_failure.csv')
preprocessing(dataset)