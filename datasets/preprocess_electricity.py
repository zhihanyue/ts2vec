import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='LD2011_2014.txt')
    parser.add_argument('-o', '--output', type=str, default='electricity.csv')
    args = parser.parse_args()
    data_ecl = pd.read_csv(args.input, parse_dates=True, sep=';', decimal=',', index_col=0)
    data_ecl = data_ecl.resample('1h', closed='right').sum()
    data_ecl = data_ecl.loc[:, data_ecl.cumsum(axis=0).iloc[8920] != 0]  # filter out instances with missing values
    data_ecl.index = data_ecl.index.rename('date')
    data_ecl = data_ecl['2012':]
    data_ecl.to_csv(args.output)
