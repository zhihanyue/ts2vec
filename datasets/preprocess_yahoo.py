import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import argparse
import pickle
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True, help='The folder of raw yahoo dataset')
    parser.add_argument('-o', '--output', type=str, default='yahoo.pkl')
    args = parser.parse_args()

    all_train_data = {}
    all_train_labels = {}
    all_train_timestamps = {}
    all_test_data = {}
    all_test_labels = {}
    all_test_timestamps = {}

    for i in range(1, 368):
        with open(os.path.join(args.path, str(i)), 'rb') as f:
            ts = pickle.load(f)
        data = np.array(ts['value'])
        labels = np.array(ts['label'])
        timestamps = np.array(ts['timestamp'])
        k = 'yahoo_' + str(i)
        l = len(data) // 2

        n = 0
        while adfuller(data[:l], 1)[1] > 0.05 or adfuller(data[:l])[1] > 0.05:
            data = np.diff(data)
            labels = labels[1:]
            timestamps = timestamps[1:]
            n += 1
        l -= n

        all_train_data[k] = data[:l]
        all_test_data[k] = data[l:]
        all_train_labels[k] = labels[:l]
        all_test_labels[k] = labels[l:]
        all_train_timestamps[k] = timestamps[:l]
        all_test_timestamps[k] = timestamps[l:]

        mean, std = all_train_data[k].mean(), all_train_data[k].std()
        all_train_data[k] = (all_train_data[k] - mean) / std
        all_test_data[k] = (all_test_data[k] - mean) / std

    with open(args.output, 'wb') as f:
        pickle.dump({
            'all_train_data': all_train_data,
            'all_train_labels': all_train_labels,
            'all_train_timestamps': all_train_timestamps,
            'all_test_data': all_test_data,
            'all_test_labels': all_test_labels,
            'all_test_timestamps': all_test_timestamps,
            'delay': 3
        }, f)
    