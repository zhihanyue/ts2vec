import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import argparse
import pickle

def _load_raw_KPI(train_filename, test_filename):
    train_data = pd.read_csv(train_filename)
    train_data = train_data.set_index(['KPI ID', 'timestamp']).sort_index()
    x_train = {}
    y_train = {}
    timestamp_train = {}
    scaler = {}
    for name, df in train_data.groupby(level=0):
        x_train[name] = df['value'].to_numpy()
        y_train[name] = df['label'].to_numpy()
        meanv = df['value'].mean()
        stdv = df['value'].std()
        scaler[name] = (meanv, stdv)
        x_train[name] = (x_train[name] - meanv) / stdv
        timestamp_train[name] = df.index.get_level_values(1)
    
    test_data = pd.read_hdf(test_filename)
    test_data['KPI ID'] = test_data['KPI ID'].apply(str)
    test_data = test_data.set_index(['KPI ID', 'timestamp']).sort_index()
    x_test = {}
    y_test = {}
    timestamp_test = {}
    for name, df in test_data.groupby(level=0):
        x_test[name] = df['value'].to_numpy()
        y_test[name] = df['label'].to_numpy()
        x_test[name] = (x_test[name] - scaler[name][0]) / scaler[name][1]
        timestamp_test[name] = df.index.get_level_values(1)
    return x_train, y_train, timestamp_train, x_test, y_test, timestamp_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-file', type=str, default='phase2_train.csv')
    parser.add_argument('--test-file', type=str, default='phase2_ground_truth.hdf')
    parser.add_argument('-o', '--output', type=str, default='kpi.pkl')
    args = parser.parse_args()

    data1, labels1, timestamps1, data2, labels2, timestamps2 = _load_raw_KPI(args.train_file, args.test_file)
    all_data = {k+'_t1': data1[k] for k in data1}
    all_data.update({k+'_t2': data2[k] for k in data2})
    all_labels = {k+'_t1': labels1[k] for k in labels1}
    all_labels.update({k+'_t2': labels2[k] for k in labels2})
    all_timestamps = {k+'_t1': timestamps1[k] for k in timestamps1}
    all_timestamps.update({k+'_t2': timestamps2[k] for k in timestamps2})
    
    all_train_data = {}
    all_train_labels = {}
    all_train_timestamps = {}
    all_test_data = {}
    all_test_labels = {}
    all_test_timestamps = {}
    
    for k in all_data:
        data = all_data[k]
        labels = all_labels[k]
        timestamps = all_timestamps[k]
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
            'delay': 7
        }, f)
    