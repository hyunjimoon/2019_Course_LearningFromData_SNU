import DataGenerator
import pickle
import numpy as np
import pandas as pd
from DataGenerator import *
from sklearn.metrics import mean_absolute_error

def main():
    ################### HYPER-PARAM ###################
    symbol_idx = [2, 14]  # 사용할 변수 설정 ##
    ''' 0: BrentOil, 1: Copper, 2: CrudeOil, 3: Gasoline, 4: Gold, 5: NaturalGas, 6: Platinum, 7: Silver,
        8: AUD, 9: CNY, 10: EUR, 11: GBP, 12: HKD, 13: JPY, 14: USD '''
    col_idx = [0]
    ''' 0: Price, 1: Open, 2: High, 3: Low, 4: Volume, 5: Change'''
    p = 0.0
    input_days = 3
    start_date = '2010-01-01'
    end_date = '2019-04-08'

    filename = 'team05_model.pkl'
    ###################################################
    symbols = symbol_info[symbol_idx]
    columns = col_info[col_idx]

    print('Test ...')
    scaled_test_x, scaled_test_y, test_y, scaler_y = make_features(start_date, end_date, symbols, columns, input_days,
                                                                   False, 0.0)
    # inspect test data
    assert test_y.tolist() == get_test_dollar_price(start_date, end_date).tolist(), 'your test data is wrong!'

    loaded_models = pickle.load(open(filename, 'rb'))
    print('...... load complete')
    #     print('...... parameter :' + str(loaded_models[0].get_params()))

    predict = np.average([loaded_models[i].predict(scaled_test_x) for i in range(len(loaded_models))], axis=0)

    print('... Test finished\n\n')

    print('FINAL RESULT')
    print('MAE : ', mean_absolute_error(scaler_y.inverse_transform(predict),
                                        scaler_y.inverse_transform(scaled_test_y)))

if __name__ == '__main__':
    main()