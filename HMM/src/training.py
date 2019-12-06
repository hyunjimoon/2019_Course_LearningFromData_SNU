from hmmlearn.hmm import GaussianHMM, GMMHMM
import pickle
import DataGenerator
import numpy as np
from sklearn.metrics import mean_absolute_error


def main():

    start_date = '2010-01-01'
    end_date = '2019-04-10'
    scaler = None  # None, 'minmax', 'standard'
    diff_symbols = ['Gold','Silver']
    price_symbols = []
    n_so = []

    input_days = [1, 1, 1, 3, 3, 3]
    n_mix = [3, 3, 3, 3, 3, 3]  # 1 : GaussianHMM과 동일
    n_mva = [[], [], [], [5], [5], [5]]
    n_roc = [[], [], [], [3], [3], [3]]


    models = list()
    for i in range(len(n_mix)):
        bm, bmae = None, 100.0
        for j in range(10):
            training_x = DataGenerator.make_features(start_date, end_date, input_days=input_days[i], scaler=scaler,
                                                 is_training=True, diff_symbols=diff_symbols, price_symbols=price_symbols,
                                                 n_mva=n_mva[i], n_roc=n_roc[i], n_so=n_so)

            model = GMMHMM(n_components=n_mix[i]*3, n_mix=n_mix[i], n_iter=1000)
            model.fit(training_x)
            mae_tmp = 0.0
            for e in ['2019-04-10', '2019-04-06', '2019-03-31', '2019-03-26', '2019-03-21']:
                test_x, past_price, target_price = DataGenerator.make_features(start_date, e, input_days=input_days[i],
                                                                               scaler=scaler, is_training=False,
                                                                               diff_symbols=diff_symbols,
                                                                               price_symbols=price_symbols,
                                                                               n_mva=n_mva[i], n_roc=n_roc[i], n_so=n_so)

                hidden_states = model.predict(test_x)
                means = np.sum(model.means_, axis=1)
                expected_diff_price = np.dot(model.transmat_, means)
                diff = list(zip(*expected_diff_price))[0]

                predicted_price = list()
                for idx in range(10):  # predict gold price for 10 days
                    state = hidden_states[idx]
                    current_price = past_price[idx]
                    next_day_price = current_price + diff[state]  # predicted gold price of next day
                    predicted_price.append(next_day_price)

                predict_tmp = np.array(predicted_price)
                mae_tmp += mean_absolute_error(target_price, predict_tmp)
            print(mae_tmp)

            if bmae > mae_tmp:
                bm, bmae = model, mae_tmp

        models.append(bm)
        print()

    filename = 'team05_model.pkl'
    pickle.dump(models, open(filename, 'wb'))
    print('saved {}'.format(filename))



if __name__ == "__main__":
    main()



