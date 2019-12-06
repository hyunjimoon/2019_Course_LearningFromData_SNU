from sklearn import neural_network as NN
import pickle
import numpy as np
import os
from DataGenerator import *


def main():
    ################### HYPER-PARAM ###################
    symbol_idx = [2, 14]  # 사용할 변수 설정 ##
    ''' 0: BrentOil, 1: Copper, 2: CrudeOil, 3: Gasoline, 4: Gold, 5: NaturalGas, 6: Platinum, 7: Silver,
        8: AUD, 9: CNY, 10: EUR, 11: GBP, 12: HKD, 13: JPY, 14: USD '''
    col_idx = [0]
    ''' 0: Price, 1: Open, 2: High, 3: Low, 4: Volume, 5: Change'''
    p = 0.0
    input_days = 3
    model_num = 7  # 앙상블할 모델의 개수
    hidden = (32,32)
    l_rate = 0.01
    start_date = '2010-01-01'
    end_date = '2019-04-08'

    filename = 'team05_model.pkl'
    is_training = True  # 참일 때는 학습+테스트 모두 수행. 거짓일 때는 테스트만 수행.
    ###################################################
    symbols = symbol_info[symbol_idx]
    columns = col_info[col_idx]
    if os.path.exists(filename):
        print(filename, 'exists')
        is_training = False

    if is_training:
        print('Training ...')
        training_x, training_y, scaler_y = make_features(start_date, end_date, symbols, columns, input_days, True, 0.0)

        models = list()
        for i in range(model_num):
            # print('...... ensemble {}'.format(i + 1))
            model_temp = NN.MLPRegressor(hidden_layer_sizes=hidden, activation='relu',
                                         solver='adam', alpha=0.0001, batch_size='auto',
                                         learning_rate='constant', learning_rate_init=l_rate,
                                         power_t=0.5, max_iter=5000, shuffle=True, random_state=i,
                                         tol=0.0001, verbose=True, warm_start=False, momentum=0.9,
                                         nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                                         beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model_temp.fit(training_x, training_y)
            models.append(model_temp)
        pickle.dump(models, open(filename, 'wb'))
        print('...... saved {}'.format(filename))
        print('... Training finished\n\n')

    else:
        print('Test ...')
        scaled_test_x, scaled_test_y, test_y, scaler_y = make_features(start_date, end_date, symbols, columns,
                                                                       input_days,
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





