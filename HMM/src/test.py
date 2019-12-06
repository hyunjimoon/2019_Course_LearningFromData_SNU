from sklearn.metrics import mean_absolute_error
import numpy as np
import pickle
import DataGenerator
import pandas as pd
from DataGenerator import get_data_path


def get_past_price(start_date, end_date):
    """
    Do not fix this function
    """
    df = pd.read_csv(get_data_path('Gold'), index_col="Date", parse_dates=True, na_values=['nan'])
    price = df['Price'].loc[end_date: start_date][1:11][::-1]
    return price


def get_target_price(start_date, end_date):
    """
    Do not fix this function
    """
    df = pd.read_csv(get_data_path('Gold'), index_col="Date", parse_dates=True, na_values=['nan'])
    price = df['Price'].loc[end_date: start_date][:10][::-1]
    return price


def main():

    start_date = '2010-01-01'
    end_date = '2019-04-10'

    scaler = None
    diff_symbols = ['Gold','Silver']
    price_symbols = []
    n_so = []

    input_days = [1, 1, 1, 3, 3, 3]
    n_mix = [3, 3, 3, 3, 3, 3]  # 1 : GaussianHMM과 동일
    n_mva = [[], [], [], [5], [5], [5]]
    n_roc = [[], [], [], [3], [3], [3]]
    ######################################

    predict = list()
    filename = 'team05_model.pkl'
    models = pickle.load(open(filename, 'rb'))
    print('load complete')
    # for end_date in ['2019-04-10', '2019-04-06', '2019-03-31', '2019-03-26', '2019-03-21']:
    #     print('\n\n', end_date)
    for i in range(6):
        test_x, past_price, target_price = DataGenerator.make_features(start_date, end_date, input_days=input_days[i],
                                                                       scaler=scaler, is_training=False,
                                                                       diff_symbols=diff_symbols,
                                                                       price_symbols=price_symbols,
                                                                       n_mva=n_mva[i], n_roc=n_roc[i], n_so=n_so)

        ###################################################################################################################
        # inspect data
        assert past_price.tolist() == get_past_price(start_date, end_date).tolist(), 'your past price data is wrong!'
        assert target_price.tolist() == get_target_price(start_date, end_date).tolist(), 'your target price data is wrong!'
        ###################################################################################################################

        hidden_states = models[i].predict(test_x)
        means = np.sum(models[i].means_, axis=1)
        expected_diff_price = np.dot(models[i].transmat_, means)
        diff = list(zip(*expected_diff_price))[0]

        predicted_price = list()
        for idx in range(10):  # predict gold price for 10 days
            state = hidden_states[idx]
            current_price = past_price[idx]
            next_day_price = current_price + diff[state]  # predicted gold price of next day
            predicted_price.append(next_day_price)

        predict_tmp = np.array(predicted_price)
        predict.append(predict_tmp)
    
    predict = np.mean(predict, axis=0)

    # print predicted_prices
    print('past price : {}'.format(np.array(past_price)))
    print('predicted price : {}'.format(predict))
    print('real price : {}'.format(np.array(target_price)))
    print()
    print('mae :', mean_absolute_error(target_price, predict))




if __name__ == '__main__':
    main()