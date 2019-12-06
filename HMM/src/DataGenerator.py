import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def get_data_path(symbol):
    # Return CSV file path given symbol.
    commodity_dir = '../data/commodities'
    currency_dir = '../data/currencies'

    if symbol in ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']:
        path = os.path.join(currency_dir, symbol + '.csv')
    else:
        path = os.path.join(commodity_dir, symbol + '.csv')

    return path


def merge_data(start_date, end_date, symbols):
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    if 'Gold' not in symbols:
        symbols.insert(0, 'Gold')

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
        df = df.join(df_temp)

    return df


def make_features(start_date, end_date, input_days, is_training, diff_symbols, price_symbols, n_mva=list(),
                  n_roc=list(), n_so=list(), scaler=None):
    # TODO: select symbols
    # commodity : BrentOil, Copper, CrudeOil, Gasoline, Gold, NaturalGas, Platinum, Silver
    # currency : AUD, CNY, EUR, GBP, HKD, JPY, USD

    table = merge_data(start_date, end_date, symbols=['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD', 'BrentOil',
                                                      'Copper', 'CrudeOil', 'Gasoline', 'Gold', 'NaturalGas',
                                                      'Platinum', 'Silver'])
    all_symbols = ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD', 'BrentOil', 'Copper', 'CrudeOil', 'Gasoline',
                   'Gold', 'NaturalGas', 'Platinum', 'Silver']

    # TODO: cleaning or filling missing value
    columns = list(table)

    for i in columns:
        table[i] = table[i].interpolate(method='linear')

    table.dropna(inplace=True)
    # print(table)

    # TODO:  select columns to use
    price = dict()
    diff = dict()
    for s in all_symbols:
        price[s] = table[s + '_Price']
        diff[s] = [0] + np.diff(price[s])
        price[s] = price[s]

    mva = dict()
    for n in n_mva:
        mva[n] = price['Gold'].rolling(n).mean().fillna(1000.0)

    days = len(price['Gold'])
    roc = dict()
    for n in n_roc:
        p = price['Gold'].rolling(5).mean().fillna(1000.0)
        roc_tmp = list()
        for i in range(days):
            if i < n:
                roc_tmp.append((price['Gold'][i] - p[0]) / p[0])
            else:
                roc_tmp.append((price['Gold'][i] - p[i - n]) / p[i - n])
        roc[n] = roc_tmp

    so = dict()
    for n in n_so:
        so_tmp = list()
        for i in range(days):
            if i < n:
                so_tmp.append(0.0)
            else:
                high = max(price['Gold'][i - n:i])
                low = min(price['Gold'][i - n:i])
                if high == low:
                    so_tmp.append(0.0)
                else:
                    so_tmp.append((price['Gold'][i] - low) / (high - low))
        so[n] = so_tmp

    training_sets = list()
    n_max = max([1] + n_mva + n_roc + n_so)
    for time in range(len(price['Gold']) - input_days - n_max):
        diff_tmp = list()
        price_tmp = list()
        mva_tmp = list()
        roc_tmp = list()
        so_tmp = list()
        for s in diff_symbols:
            diff_tmp.append(diff[s][time + n_max:time + input_days + n_max][::-1])
        for s in price_symbols:
            price_tmp.append(diff[s][time + n_max:time + input_days + n_max][::-1])
        for n in n_mva:
            mva_tmp.append(mva[n][time + n_max:time + input_days + n_max][::-1])
        for n in n_roc:
            roc_tmp.append(roc[n][time + n_max:time + input_days + n_max][::-1])
        for n in n_so:
            so_tmp.append(so[n][time + n_max:time + input_days + n_max][::-1])

        daily_feature = np.concatenate(diff_tmp + price_tmp + mva_tmp + roc_tmp + so_tmp, axis=0)
        training_sets.append(daily_feature)

    training_x = training_sets[:-10]
    test_x = training_sets[-10:]

    if scaler == 'minmax':
        scaler = StandardScaler()
        training_x = scaler.fit_transform(training_x)
        test_x = scaler.transform(test_x)
    elif scaler == 'standard':
        scaler = MinMaxScaler()
        training_x = scaler.fit_transform(training_x)
        test_x = scaler.transform(test_x)

    past_price = price['Gold'][-11:-1]
    target_price = price['Gold'][-10:]

    return training_x if is_training else (test_x, past_price, target_price)

