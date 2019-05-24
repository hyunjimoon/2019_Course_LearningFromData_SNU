import os
import pandas as pd
import numpy as np

DATA_PATH = '../data/'

column_name = {
    'race_result': ['date', 'race_num', 'track_length', 'track_state', 'weather', 'rank', 'lane', 'horse', 'home',
                    'gender', 'age', 'weight', 'rating', 'jockey', 'trainer', 'owner', 'single_odds', 'double_odds'],
    'jockey': ['jockey', 'group', 'birth', 'age', 'debut', 'weight', 'weight_2', 'race_count', 'first', 'second',
               '1yr_count', '1yr_first', '1yr_second'],
    'owner': ['owner', 'reg_horse', 'unreg_horse', 'owned_horse', 'reg_date', '1yr_count', '1yr_first', '1yr_second',
              '1yr_third', '1yr_money', 'race_count', 'first', 'second', 'third', 'owner_money'],
    'trainer': ['trainer', 'group', 'birth', 'age', 'debut', 'race_count', 'first', 'second', '1yr_count', '1yr_first',
                '1yr_second'],
    'horse': ['horse', 'home', 'gender', 'birth', 'age', 'class', 'group', 'trainer', 'owner', 'father', 'mother',
              'race_count', 'first', 'second', '1yr_count', '1yr_first', '1yr_second', 'horse_money', 'rating',
              'price'],
}

# TODO: select columns to use
used_column_name = {
    'race_result': ['date', 'race_num', 'rank', 'lane', 'horse', 'jockey', 'trainer', 'owner'],
    'jockey': ['date', 'jockey', 'weight'],
    'owner': ['date', 'owner', 'owner_money'],
    'trainer': ['date', 'trainer', 'race_count'],
    'horse': ['date', 'horse', 'age', '1yr_first', '1yr_second', 'price'],
}


def load_data():
    df_dict = dict()  # key: data type(e.g. jockey, trainer, ...), value: corresponding dataframe

    for data_type in ['horse', 'jockey', 'owner', 'trainer', 'race_result']:
        fnames = sorted(os.listdir(DATA_PATH + data_type))

        df = pd.DataFrame()

        # concatenate all text files in the directory
        for fname in fnames:
            tmp = pd.read_csv(os.path.join(DATA_PATH, data_type, fname), header=None, sep=",",
                              encoding='cp949', names=column_name[data_type])

            if data_type != 'race_result':
                date = fname.split('.')[0]
                tmp['date'] = date[:4] + "-" + date[4:6] + "-" + date[-2:]

            df = pd.concat([df, tmp])

        # cast date column to dtype datetime
        df['date'] = df['date'].astype('datetime64[ns]')

        # append date offset to synchronize date with date of race_result data
        if data_type != 'race_result':
            df1 = df.copy()
            df1['date'] += pd.DateOffset(days=2)  # saturday
            df2 = df.copy()
            df2['date'] += pd.DateOffset(days=3)  # sunday
            df = df1.append(df2)

        # select columns to use
        df = df[used_column_name[data_type]]

        # insert dataframe to dictionary
        df_dict[data_type] = df
## race result에 모든 날짜 맞추겠음
    df_dict['race_result']['rank'].replace('1', 1., inplace=True)
    df_dict['race_result']['rank'].replace('2', 2., inplace=True)
    df_dict['race_result']['rank'].replace('3', 3., inplace=True)
    df_dict['race_result']['rank'].replace('4', 4., inplace=True)
    df_dict['race_result']['rank'].replace('5', 5., inplace=True)
    df_dict['race_result']['rank'].replace('6', 6., inplace=True)
    df_dict['race_result']['rank'].replace('7', 7., inplace=True)
    df_dict['race_result']['rank'].replace('8', 8., inplace=True)
    df_dict['race_result']['rank'].replace('9', 9., inplace=True)
    df_dict['race_result']['rank'].replace('10', 10., inplace=True)
    df_dict['race_result']['rank'].replace('11', 11., inplace=True)
    df_dict['race_result']['rank'].replace('12', 12., inplace=True)
    df_dict['race_result']['rank'].replace('13', 13., inplace=True)
    df_dict['race_result']['rank'].replace(' ', np.nan, inplace=True)
    
    # drop rows with rank missing values
    df_dict['race_result'].dropna(subset=['rank'], inplace=True)

    df_dict['race_result']['rank'] = df_dict['race_result']['rank'].astype('int')
    # make a column 'win' that indicates whether a horse ranked within the 3rd place
    df_dict['race_result']['win'] = df_dict['race_result'].apply(lambda x: 1 if x['rank'] < 4 else 0, axis=1)

    # drop duplicated rows
    df_dict['jockey'].drop_duplicates(subset=['date', 'jockey'], inplace=True)
    df_dict['owner'].drop_duplicates(subset=['date', 'owner'], inplace=True)
    df_dict['trainer'].drop_duplicates(subset=['date', 'trainer'], inplace=True)

    # merge dataframes
    df = df_dict['race_result'].merge(df_dict['horse'], on=['date', 'horse'], how='left')
    df = df.merge(df_dict['jockey'], on=['date', 'jockey'], how='left')
    df = df.merge(df_dict['owner'], on=['date', 'owner'], how='left')
    df = df.merge(df_dict['trainer'], on=['date', 'trainer'], how='left')

    # drop unnecessary columns which are used only for merging dataframes
    df.drop(['horse', 'jockey', 'trainer', 'owner'], axis=1, inplace=True)

    return df


def get_data(test_day, is_training):
    data_set = load_data()

    # select training and test data by test day
    # TODO : cleaning or filling missing value
    training_data = data_set[~data_set['date'].isin(test_day)].fillna(0)
    test_data = data_set[data_set['date'].isin(test_day)].fillna(0)

    # TODO : make your input feature columns

    # select training x and y
    training_y = training_data['win']
    training_x = training_data.drop(['win', 'date', 'race_num', 'rank', 'lane'], axis=1)

    # select test x and y
    ## 몇 번 레인에서 뛰는게 중요한 경우 넣어도 됨(일단 현재는 빼놓음)
    test_y = test_data['win']
    test_x = test_data.drop(['win', 'date', 'race_num', 'rank', 'lane'], axis=1)

    inspect_test_data(test_x, test_day)

    return (training_x, training_y) if is_training else (test_x, test_y)


def inspect_test_data(test_x, test_days):
    df = pd.DataFrame()

    for test_day in test_days:
        fname = os.path.join(DATA_PATH, 'race_result', test_day.replace('-', '') + '.csv')
        tmp = pd.read_csv(fname, header=None, sep=",",
                          encoding='cp949', names=column_name['race_result'])
        tmp.replace(' ', np.nan, inplace=True)
        tmp.dropna(subset=['rank'], inplace=True)

        df = pd.concat([df, tmp])

    print(test_x.shape[0])
    print(df.shape[0])

    assert test_x.shape[0] == df.shape[0], 'your test data is wrong!'


def main():
    get_data(['2019-04-20', '2019-04-21'], is_training=True)


if __name__ == '__main__':
    main()
