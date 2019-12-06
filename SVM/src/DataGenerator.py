import os
import pandas as pd
import numpy as np

DATA_PATH = '../data/'

column_name = {
    'race_result': ['date', 'race_num', 'track_length', 'track_state', 'weather', 'rank', 'lane', 'horse', 'home',
                    'gender', 'age', 'weight', 'rating', 'jockey', 'trainer', 'owner', 'single_odds', 'double_odds'],
    'jockey': ['jockey', 'group', 'birth', 'age', 'debut', 'weight', 'weight_2',
               'jockey_race_count', 'jockey_first', 'jockey_second',
               'jockey_1yr_count', 'jockey_1yr_first', 'jockey_1yr_second'],
    'owner': ['owner', 'reg_horse', 'unreg_horse', 'owned_horse', 'reg_date',
              '1yr_count', '1yr_first', '1yr_second', '1yr_third', '1yr_money',
              'race_count', 'first', 'second', 'third', 'owner_money'],
    'trainer': ['trainer', 'group', 'birth', 'age', 'debut',
                'trainer_race_count', 'trainer_first', 'trainer_second',
                'trainer_1yr_count', 'trainer_1yr_first', 'trainer_1yr_second'],
    'horse': ['horse', 'home', 'gender', 'birth', 'age', 'class', 'group', 'trainer', 'owner', 'father', 'mother',
              'horse_race_count', 'horse_first', 'horse_second',
              'horse_1yr_count', 'horse_1yr_first', 'horse_1yr_second',
              'horse_money', 'rating', 'price'],
}

# TODO: select columns to use

used_column_name = {
    'race_result': ['date', 'race_num', 'rank', 'lane', 'horse', 'home', 'gender', 'weight',
                    'jockey', 'trainer', 'owner'],
    'jockey': ['date', 'jockey', 'jockey_1yr_count', 'jockey_1yr_first', 'jockey_1yr_second',
               'jockey_race_count', 'jockey_first', 'jockey_second'],
    'owner': ['date', 'owner', 'owner_money'],
    'trainer': ['date', 'trainer', 'trainer_1yr_count', 'trainer_1yr_first', 'trainer_1yr_second',
                'trainer_race_count', 'trainer_first', 'trainer_second'],
    'horse': ['date', 'horse', 'age', 'price', 'horse_money', 'rating',
              'horse_1yr_count', 'horse_1yr_first', 'horse_1yr_second',
              'horse_race_count', 'horse_first', 'horse_second'],
}


def load_data():
    df_dict = dict()  # key: data type(e.g. jockey, trainer, ...), value: corresponding dataframe

    for data_type in ['race_result', 'horse', 'jockey', 'owner', 'trainer']:
        fnames = sorted(os.listdir(DATA_PATH + data_type))
        df = pd.DataFrame()

        for fname in fnames:
            tmp = pd.read_csv(os.path.join(DATA_PATH, data_type, fname), header=None, sep=",",
                              encoding='cp949', names=column_name[data_type])
            if data_type != 'race_result':
                date = fname.split('.')[0]
                tmp['date'] = date[:4] + "-" + date[4:6] + "-" + date[-2:]
            else:  #####
                for i in range(tmp.shape[0]):
                    o = tmp.loc[i, 'owner']
                    if '♠' in str(o):
                        tmp.loc[i, 'owner'] = o[1:]

            df = pd.concat([df, tmp])

        df['date'] = df['date'].astype('datetime64[ns]')

        if data_type != 'race_result':
            df1 = df.copy()
            df1['date'] += pd.DateOffset(days=2)  # saturday
            df2 = df.copy()
            df2['date'] += pd.DateOffset(days=3)  # sunday
            df = df1.append(df2)

        df = df[used_column_name[data_type]]
        df_dict[data_type] = df

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
    df_dict['race_result'].dropna(subset=['rank'], inplace=True)
    df_dict['race_result']['rank'] = df_dict['race_result']['rank'].astype('int')

    df_dict['race_result']['win'] = df_dict['race_result'].apply(lambda x: 1 if x['rank'] < 4 else 0, axis=1)

    ##### dummy-variable
    df_dict['race_result']['gender'].replace({'암': 'F', '수': 'M', '거': 'G'}, inplace=True)
    df_dict['race_result'] = pd.concat([df_dict['race_result'],
                                        pd.get_dummies(df_dict['race_result']['gender'])], axis=1)
    df_dict['race_result']['home'].replace({'한': 'K'}, inplace=True)
    df_dict['race_result'] = pd.concat([df_dict['race_result'],
                                        pd.get_dummies(df_dict['race_result']['home'])[['K']]], axis=1)
    del df_dict['race_result']['gender']
    del df_dict['race_result']['home']

    ##### make winning ratio column
    df_dict['jockey']['jockey_1yr_win'] = df_dict['jockey'].apply(
        lambda x: (x['jockey_1yr_first'] + x['jockey_1yr_second']) / x['jockey_1yr_count']
        if x['jockey_1yr_count'] > 0 else 0, axis=1)
    df_dict['jockey']['jockey_total_win'] = df_dict['jockey'].apply(
        lambda x: (x['jockey_first'] + x['jockey_second']) / x['jockey_race_count']
        if x['jockey_race_count'] > 0 else 0, axis=1)

    df_dict['trainer']['trainer_1yr_win'] = df_dict['trainer'].apply(
        lambda x: (x['trainer_1yr_first'] + x['trainer_1yr_second']) / x['trainer_1yr_count']
        if x['trainer_1yr_count'] > 0 else 0, axis=1)
    df_dict['trainer']['trainer_total_win'] = df_dict['trainer'].apply(
        lambda x: (x['trainer_first'] + x['trainer_second']) / x['trainer_race_count']
        if x['trainer_race_count'] > 0 else 0, axis=1)

    df_dict['horse']['horse_1yr_win'] = df_dict['horse'].apply(
        lambda x: (x['horse_1yr_first'] + x['horse_1yr_second']) / x['horse_1yr_count']
        if x['horse_1yr_count'] > 0 else 0, axis=1)
    df_dict['horse']['horse_total_win'] = df_dict['horse'].apply(
        lambda x: (x['horse_first'] + x['horse_second']) / x['horse_race_count']
        if x['horse_race_count'] > 0 else 0, axis=1)
    df_dict['horse']['horse_money'] = df_dict['horse'].apply(
        lambda x: x['horse_money'] / x['horse_race_count'] if x['horse_race_count'] > 0 else 0, axis=1)

    # recent 3 rank mean
    dfjr = df_dict['race_result'].merge(df_dict['jockey'], on=['date', 'jockey'], how='left')
    j_rank = pd.DataFrame(dfjr.groupby(['jockey', 'date']).mean()['rank']). \
        rolling(4, min_periods=1).apply(lambda x: x[:-1].mean() if x.shape[0] - 1 else 5, raw=True)
    j_rank.rename(columns={'rank': 'j_rank'}, inplace=True)
    df_dict['race_result'] = df_dict['race_result'].merge(j_rank, on=['jockey', 'date'])

    dfhr = df_dict['race_result'].merge(df_dict['horse'], on=['date', 'horse'], how='left')
    h_rank = pd.DataFrame(dfhr.groupby(['horse', 'date']).mean()['rank']). \
        rolling(4, min_periods=1).apply(lambda x: x[:-1].mean() if x.shape[0] - 1 else 5, raw=True)
    h_rank.rename(columns={'rank': 'h_rank'}, inplace=True)
    df_dict['race_result'] = df_dict['race_result'].merge(h_rank, on=['horse', 'date'])

    # drop duplicated rows
    df_dict['jockey'].drop_duplicates(subset=['date', 'jockey'], inplace=True)
    df_dict['owner'].drop_duplicates(subset=['date', 'owner'], inplace=True)
    df_dict['trainer'].drop_duplicates(subset=['date', 'trainer'], inplace=True)

    # merge dataframes
    df = df_dict['race_result'].merge(df_dict['horse'], on=['date', 'horse'], how='left')
    df = df.merge(df_dict['jockey'], on=['date', 'jockey'], how='left')
    df = df.merge(df_dict['owner'], on=['date', 'owner'], how='left')
    df = df.merge(df_dict['trainer'], on=['date', 'trainer'], how='left')

    df.to_csv('df_final.csv')
    return df


def get_data(test_day, var_list, is_training=False):
    idx_list = ['date', 'race_num']
    unnorm_var_list = ['lane', 'F', 'M', 'G', 'K', 'age', 'weight']
    norm_var_list = [col for col in var_list if col not in unnorm_var_list]
    if os.path.exists('df_final.csv'):
        print('preprocessed data exists')
        original_data_set = pd.read_csv('df_final.csv')
    else:
        print('loading data')
        original_data_set = load_data()

    data_set = original_data_set[idx_list + var_list + ['win']]
    training_data = data_set[~data_set['date'].isin(test_day)]
    training_data = training_data.fillna(training_data.mean())
    test_data = data_set[data_set['date'].isin(test_day)]
    test_data = test_data.fillna(training_data.mean())

    if is_training:
        training_y = training_data['win']
        training_x = training_data[idx_list + var_list]
        training_x2 = (training_x[idx_list + norm_var_list].groupby(idx_list) \
                       .transform(lambda x: (x - x.mean()) / x.std())).fillna(0.0)
        training_x2.columns = ['norm_' + c for c in norm_var_list]
        training_x = training_x[var_list].transform(lambda x: (x - x.mean()) / x.std()).fillna(0.0)
        # training_x = training_x[unnorm_var_list].transform(lambda x: (x - x.mean()) / x.std()).fillna(0.0)
        training_x = pd.concat([training_x, training_x2], axis=1, sort=False)
        training_d = training_data[idx_list]

    else:
        test_y = test_data['win']
        test_x = test_data[idx_list + var_list]
        test_x2 = (test_x[idx_list + norm_var_list].groupby(idx_list) \
                   .transform(lambda x: (x - x.mean()) / x.std())).fillna(0.0)
        test_x2.columns = ['norm_' + c for c in norm_var_list]
        test_x = test_x[var_list].transform(lambda x: (x - x.mean()) / x.std()).fillna(0.0)
        # test_x = test_x[unnorm_var_list].transform(lambda x: (x - x.mean()) / x.std()).fillna(0.0)
        test_x = pd.concat([test_x, test_x2], axis=1, sort=False)
        test_d = test_data[idx_list]

        inspect_test_data(test_x, test_day)

    return (training_x, training_y, training_d) if is_training else (test_x, test_y, test_d)


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
