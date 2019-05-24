import os
import pandas as pd


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


def make_features(start_date, end_date, is_training):

    # TODO: Choose symbols to read
    # symbols = ['Celltrion', 'HyundaiMobis', 'HyundaiMotor', 'KOSPI', 'LGChemical', 'LGH&H', 'POSCO',
    # 'SamsungElectronics', 'SamsungElectronics2', 'ShinhanFinancialGroup', 'SKhynix']
    symbols = ['SamsungElectronics']

    table = merge_data(start_date, end_date, symbols)

    # TODO: select columns to use
    s_close = table['SamsungElectronics_close']
    s_open = table['SamsungElectronics_open']

    # TODO: make features
    input_days = 3

    features = list()
    for a in range(len(s_close)-input_days):
        features.append(s_close[a:a+input_days])

    s_close = s_close[input_days:]
    s_open = s_open[input_days:] ################## input_days를 자르는 이유?  앞쪽은 feature생성에만 쓰기위해

    test_days = 10
    if not is_training:
        return s_open[-test_days:], s_close[-test_days:], features[-test_days:]

    return s_open, s_close, features


if __name__ == "__main__":
    #main() ############################## 실행 안됨?
    open, close, feature = make_features('2010-01-01', '2019-05-08', False)
    print("end")