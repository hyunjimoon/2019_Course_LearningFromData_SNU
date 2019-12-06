import os
import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt

def symbol_to_path(symbol, base_dir="../data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def merge_data(start_date, end_date, symbols):
    dates = pd.date_range(start_date, end_date)

    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date", parse_dates=True,
                              usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Open': symbol + '_open', 'High': symbol + '_high', 'Low': symbol + '_low',
                                          'Close': symbol + '_close', 'Volume': symbol + '_volume'})
        df = df.join(df_temp)

    # TODO: cleaning or filling missing value
    df = df.dropna()

    return df


### functions for feature engineering ############
def ema(data, alpha=0.5):
    return data.ewm(alpha=alpha).mean()

def ema_d(data, alpha=0.5):
    ema_value = data.ewm(alpha=alpha).mean()
    ema_d_value = []
    for i in range(len(ema_value)):
        if data[i] > ema_value[i]:
            ema_d_value.append(1)
        else:
            ema_d_value.append(-1)
    return np.asarray(ema_d_value)

def sma(data, input_days):
    sma_value = data.rolling(window=input_days, min_periods=0).mean()
    return sma_value

def sma_d(data, input_days):
    sma_value = sma(data, input_days)
    sma_d_value = []
    for i in range(len(sma_value)):
        if data[i] > sma_value[i]:
            sma_d_value.append(1)
        else:
            sma_d_value.append(-1)
    return np.asarray(sma_d_value)

def bnf(data, input_days, c):  ##
    sma_value = sma(data, input_days)

    bnf_value = ((data - sma_value) / sma_value) * 100

    cond1 = bnf_value > c
    cond2 = (bnf_value >= -c) & (bnf_value <= c)
    cond3 = (bnf_value < -c)

    bnf_value.loc[cond1] = 1
    bnf_value.loc[cond2] = 0
    bnf_value.loc[cond3] = -1

    return np.asarray(bnf_value)

def tendency(data, input_days):  ##
    sma_value = sma(data, input_days)
    diff = sma_value.diff().fillna(0)

    tendency_value = (diff / (sma_value - diff))
    tendency_value.fillna(0)

    return np.asarray(tendency_value)

def hc_ratio(high, close):
    return high / close

def lc_ratio(low, close):
    return low / close

def olastc_ratio(open, close):
    close_last = [close[0]]
    for i in range(len(close)):
        close_last.append(close[i])
    close_last = close_last[:-1]

    return open/close_last

def volume_ratio(volume):
    volume_last = [volume[0]]
    for i in range(len(volume)):
        volume_last.append(volume[i])
    volume_last = volume_last[:-1]

    return volume/volume_last

def ma5_close(close):
    close_ma5 = []
    for i in range(len(close)):
        if i < 5:
            close_ma5.append(np.mean(close[:i+1]))
        else:
            close_ma5.append(np.mean(close[i-4:i+1]))
    return close/close_ma5

def ma5_volume(volume):
    volume_ma5 = []
    for i in range(len(volume)):
        if i < 5:
            volume_ma5.append(np.mean(volume[:i+1]))
        else:
            volume_ma5.append(np.mean(volume[i-4:i+1]))
    return volume/volume_ma5

def macd(data):
    macd_value = (data.ewm(span=12,min_periods=0, adjust=True, ignore_na=False).mean() -
                  data.ewm(span=26,min_periods=0, adjust=True, ignore_na=False).mean())
    return np.asarray(macd_value)

def macd_d(data):
    macd_value = macd(data)
    macd_d_value = []
    for i in range(len(macd_value)):
        if macd_value[i] > macd_value[i-1]:
            macd_d_value.append(1)
        else:
            macd_d_value.append(-1)
    return np.asarray(macd_d_value)

def momentum(data, n):
    momentum_value = []
    for i in range(len(data)):
        if i < 10:
            momentum_value.append(100)
        else:
            momentum_value.append(data.iloc[i]*100/data.iloc[i-n])
    return np.asarray(momentum_value)

##############################

def make_features(start_date, end_date, company_list, is_training):

    test_days = 10

    input_days = 5 * 1  # Q가 shallow해서 적당히
    # input_days = 5 * 52 # Q가 shallow해서 적당히
    # input_days = 1 # Q가 shallow해서 적당히

    # symbols = ['Celltrion', 'HyundaiMobis', 'HyundaiMotor', 'KOSPI', 'LGChemical', 'LGH&H', 'POSCO',
    #            'SamsungElectronics', 'SamsungElectronics2', 'ShinhanFinancialGroup', 'SKhynix']
    symbols = ['Celltrion', 'KOSPI', 'LGChemical', 'LGH&H', 'SamsungElectronics', 'SamsungElectronics2', 'SKhynix']
    # symbols = ['Celltrion', 'LGChemical', 'LGH&H', 'SamsungElectronics', 'SKhynix', 'KOSPI']
    symbol_dict = {'c': 'Celltrion',
                   'lgchem': 'LGChemical',
                   'lgnh': 'LGH&H',
                   's': 'SamsungElectronics',
                   'sk':'SKhynix',
                   'k': 'KOSPI',
                   'hmobis': 'HyundaiMobis',
                   'hmotor': 'HyundaiMotor',
                   'p': 'POSCO',
                   's2': 'SamsungElectronics2',
                   'sh': 'ShinhanFinancialGroup'}

    table = merge_data(start_date, end_date, symbols)

    window_size = 25
    data = dict()
    for company in company_list:
        data[company, 'close'] = table[symbol_dict[company] + '_close']
        data[company, 'open'] = table[symbol_dict[company] + '_open']
        data[company, 'ema_open'] = ema(data[company, 'open'])  ###
        data[company, 'sma_open'] = sma(data[company, 'open'], window_size)  ###

        data[company, 'high'] = table[symbol_dict[company] + '_high']
        data[company, 'low'] = table[symbol_dict[company] + '_low']
        data[company, 'volume'] = table[symbol_dict[company] + '_volume']

        data[company, 'ema_d_open'] = ema_d(data[company, 'open'])
        data[company, 'sma_d_open'] = sma_d(data[company, 'open'], window_size)
        # data[company, 'bnf_open'] = bnf(data[company, 'open'], 25, 5)  ##
        data[company, 'bnf_open'] = bnf(data[company, 'open'], 5, 1)  ##
        data[company, 'tendency_open'] = tendency(data[company, 'open'], 25)
        data[company, 'hc_ratio'] = hc_ratio(data[company, 'high'], data[company, 'close'])
        data[company, 'lc_ratio'] = lc_ratio(data[company, 'low'], data[company, 'close'])
        data[company, 'olastc_ratio'] = olastc_ratio(data[company, 'open'], data[company, 'close'])
        data[company, 'volume_ratio'] = volume_ratio(data[company, 'volume'])
        data[company, 'ma5_close'] = ma5_close(data[company, 'close'])
        data[company, 'ma5_volume'] = ma5_volume(data[company, 'volume'])
        data[company, 'macd_open'] = macd(data[company, 'open'])
        data[company, 'macd_d_open'] = macd_d(data[company, 'open'])
        data[company, 'momentum_open'] = momentum(data[company, 'open'], 14)

    data['k', 'close'] = table[symbol_dict['k'] + '_close']
    # for d in data:
    #     if '_d_' in d[1]:
    #         continue
    #     plt.title('{}'.format(d))
    #     plt.plot(data[d])
    #     plt.show()
    # TODO: make features

    tmps = list()
    for company in company_list:
        tmp = data[company, 'open']  # 시가 기준 정규화
        tmps.append(tmp)
    tmps = np.asarray(tmps)
    open_prices = tmps.T

    tmps = list()
    for company in company_list:
        tmp = data[company, 'close']  # 시가 기준 정규화
        tmps.append(tmp)
    tmps = np.asarray(tmps)
    close_prices = tmps.T


    features = list()
    for a in range(data['k', 'close'].shape[0] - input_days):

        k_feature = data['k', 'close'][a:a + input_days]  # kospi 추가 및 정규화
        k_feature = k_feature / k_feature[-1] - 1
        k_feature = k_feature[:-1]

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'close'][a:a + input_days]  # 시가 기준 정규화
            tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            tmp = tmp[:-1]  # abandon unncessary: which is 1
            tmps.append(tmp)
        close_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'ema_open'][a:a + input_days]
            tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            tmp = tmp[:-1]
            tmps.append(tmp)
        ema_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'sma_open'][a:a + input_days]
            tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            tmp = tmp[:-1]
            tmps.append(tmp)
        sma_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'ema_d_open'][a:a + input_days]
            tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            tmp = tmp[:-1]
            tmps.append(tmp)
        ema_d_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'sma_d_open'][a:a + input_days]
            tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            tmp = tmp[:-1]
            tmps.append(tmp)
        sma_d_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'bnf_open'][a:a + input_days]
            # tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            # tmp = tmp[:-1]
            tmps.append(tmp)
        bnf_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'tendency_open'][a:a + input_days]
            # tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            # tmp = tmp[:-1]
            tmps.append(tmp)
        tendency_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'hc_ratio'][a:a + input_days]
            # tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            # tmp = tmp[:-1]
            tmps.append(tmp)
        hc_ratio_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'lc_ratio'][a:a + input_days]
            # tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            # tmp = tmp[:-1]
            tmps.append(tmp)
        lc_ratio_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'olastc_ratio'][a:a + input_days]
            # tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            # tmp = tmp[:-1]
            tmps.append(tmp)
        olastc_ratio_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'volume_ratio'][a:a + input_days]
            # tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            # tmp = tmp[:-1]
            tmps.append(tmp)
        volume_ratio_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'ma5_close'][a:a + input_days]
            tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            tmp = tmp[:-1]
            tmps.append(tmp)
        ma5_close_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'ma5_volume'][a:a + input_days]
            tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            tmp = tmp[:-1]
            tmps.append(tmp)
        ma5_volume_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'macd_open'][a:a + input_days]
            tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            tmp = tmp[:-1]
            tmps.append(tmp)
        macd_open_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'macd_d_open'][a:a + input_days]
            tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            tmp = tmp[:-1]
            tmps.append(tmp)
        macd_d_open_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for company in company_list:
            if company == 'k':
                continue
            tmp = data[company, 'momentum_open'][a:a + input_days]
            tmp = tmp / tmp[-1] - 1  # 스케일링 위해 3개 값 중 마지막으로 앞 두개 값 나눠줌
            tmp = tmp[:-1]
            tmps.append(tmp)
        momentum_open_feature = np.concatenate(tmps, axis=0)

        ### 기업 4개
        features.append(np.concatenate([
                                        close_feature,
                                        k_feature,
                                        ema_feature,
                                        sma_feature,
                                        ma5_close_feature,
                                        momentum_open_feature
                                        ], axis=0))
        ## 기업 6개
        # features.append(np.concatenate([
        #                                 close_feature,
        #                                 ema_feature,
        #                                 ma5_close_feature,
        #                                 momentum_open_feature
        #                                 ], axis=0))

    print("Load Finished")

    if not is_training:
        return open_prices[-test_days:], \
               close_prices[-test_days:], \
               features[-test_days:]

    return open_prices[input_days:], \
           close_prices[input_days:], \
           features
