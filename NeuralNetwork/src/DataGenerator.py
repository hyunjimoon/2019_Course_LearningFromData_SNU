import os
import pandas as pd
import numpy as np
import pickle
from sklearn import neural_network as NN
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

#######################################
symbol_info = np.asarray(['BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 'Gold', 'NaturalGas', 'Platinum', 'Silver',
                          'AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD'])

var_info = {'BrentOil': ['Price', 'Open', 'High', 'Low', 'Volume', 'Change'],
            'Copper' : ['Price', 'Open', 'High', 'Low', 'Volume', 'Change'],
            'CrudeOil' : ['Price', 'Open', 'High', 'Low', 'Volume', 'Change'],
            'Gasoline' : ['Price', 'Open', 'High', 'Low', 'Volume', 'Change'],
            'Gold' : ['Price', 'Open', 'High', 'Low', 'Volume', 'Change'],
            'NaturalGas' : ['Price', 'Open', 'High', 'Low', 'Volume', 'Change'],
            'Platinum' : ['Price', 'Open', 'High', 'Low', 'Change'], # Volume에 아무 값도 없음
            'Silver' : ['Price', 'Open', 'High', 'Low', 'Volume', 'Change'],
            'AUD' : ['Price', 'Open', 'High', 'Low', 'Change'],
            'CNY' : ['Price', 'Open', 'High', 'Low', 'Change'],
            'EUR' : ['Price', 'Open', 'High', 'Low', 'Change'],
            'GBP' : ['Price', 'Open', 'High', 'Low', 'Change'],
            'HKD' : ['Price', 'Open', 'High', 'Low', 'Change'],
            'JPY' : ['Price', 'Open', 'High', 'Low', 'Change'],
            'USD' : ['Price', 'Open', 'High', 'Low', 'Change'] }

col_info = np.asarray(['Price', 'Open', 'High', 'Low', 'Volume', 'Change'])
#######################################

def get_test_dollar_price(start_date, end_date):
    """
    Do not fix this function
    """
    df = pd.read_csv(get_data_path('USD'), index_col="Date", parse_dates=True, na_values=['nan'])
    price = df['Price'].loc[end_date: start_date][:10][::-1]
    return price


def get_data_path(symbol):
    # Return CSV file path given symbol.
    commodity_dir = '../data/commodities'
    currency_dir = '../data/currencies'

    if symbol in ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']:
        path = os.path.join(currency_dir, symbol + '.csv')
    else:
        path = os.path.join(commodity_dir, symbol + '.csv')

    return path


def merge_data(start_date, end_date, symbols, p):
    dates = pd.date_range(start_date, end_date, freq='D')
    dates = dates[dates.weekday != 5]
    df = pd.DataFrame(index=dates)
    if 'USD' not in symbols:
        symbols.insert(0, 'USD')

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp = df_temp.reindex(dates)
        df_temp = df_temp[var_info[symbol]]
        df_temp.columns = [symbol + '_' + v for v in var_info[symbol]]  # rename columns
        if symbol in ['BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 'Gold', 'NaturalGas', 'Silver']:
            col = symbol + '_Volume'
            df_temp[col] = pd.to_numeric(df_temp[col].str.slice(0, -1, 1), downcast='float')
        df = df.join(df_temp)
    df1 = df.fillna(method='ffill', limit=3).fillna(method='bfill', limit=3)
    df2 = df.fillna(method='bfill', limit=3).fillna(method='ffill', limit=3)
    df = df1 * p + df2 * (1 - p)
    USD_idx = pd.read_csv(get_data_path('USD'), index_col="Date", parse_dates=True,
                          na_values=['nan']).index.sort_values()

    return df.loc[:USD_idx[-1]]


def make_features(start_date, end_date, symbols, cols, input_days, is_training, p=0.0):
    data = merge_data(start_date, end_date, symbols, p)

    if 'Price' not in cols:
        cols.insert(0, 'Price')

    # TODO: select columns to use
    columns = list()
    for c in data.columns:
        if c.split("_")[1] in cols:
            columns.append(c)
    data = data[columns]
    USD_price = data['USD_Price']
    x = windowing_x(data, input_days)
    y = windowing_y(USD_price, input_days)

    # split training and test data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    training_x = x[:-10]
    training_y = np.asarray(y[:-10])
    scaled_training_x = scaler_x.fit_transform(training_x)
    scaled_training_y = scaler_y.fit_transform(training_y)
    test_x = np.reshape(x[-10], [1, -1])
    test_y = np.asarray([y[-10]])
    scaled_test_x = scaler_x.transform(test_x)
    scaled_test_y = scaler_y.transform(test_y)

    #     print('...... shape of training x: {}'.format(training_x.shape))
    #     print('...... shape of test x: {}'.format(test_x.shape))
    #     print('...... shape of training y: {}'.format(training_y.shape))
    #     print('...... shape of test y: {}\n'.format(test_y.shape))

    return (scaled_training_x, scaled_training_y, scaler_y) if is_training \
        else (scaled_test_x, scaled_test_y, test_y[0], scaler_y)


def windowing_y(data, input_days):
    input_size = len(data) - input_days
    windows = [data.iloc[i + input_days: i + input_days + 10] for i in range(input_size)]
    return windows


def windowing_x(data, input_days):
    input_size = len(data) - input_days
    windows_day = np.zeros((input_size, 7))
    windows_day[np.arange(input_size), data.index[0:input_size].dayofweek] = 1.0
    data = np.asarray(data)
    windows = np.asarray([np.concatenate([windows_day[i],
                                          np.reshape(data[i:i + input_days], (-1))], axis=0)
                          for i in range(input_size)])
    return windows

