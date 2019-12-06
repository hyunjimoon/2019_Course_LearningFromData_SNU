from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pickle
import DataGenerator


def main():
    var_list = ['lane', 'F', 'M', 'G', 'K', 'age', 'weight', 'price', 'horse_money', 'rating',
                'horse_1yr_win', 'horse_total_win', 'jockey_1yr_win', 'jockey_total_win',
                'j_rank', 'h_rank']
    param_list = [('rbf', 0.8)] * 3 + [('linear', 1.0)] * 6  # (kernel, C, classweight)
    N = 1000
    test_day = ['2019-04-13', '2019-04-14', '2019-04-20', '2019-04-21']

    training_x, training_y, _ = DataGenerator.get_data(test_day, var_list, is_training=True)

    # ================================ train SVM model=========================================

    print('start training model')

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, class_weight='balanced').fit(training_x, training_y)
    model_select = SelectFromModel(lsvc, prefit=True)
    training_x = model_select.transform(training_x)
    training_x0 = training_x[training_y == 0]
    training_x1 = training_x[training_y == 1]
    rand_idx0 = np.arange(training_x0.shape[0])
    rand_idx1 = np.arange(training_x1.shape[0])
    models = list()
    for i, (kernel, C) in enumerate(param_list):
        np.random.shuffle(rand_idx0)
        x0 = training_x0[rand_idx0[:N]]
        np.random.shuffle(rand_idx1)
        x1 = training_x1[rand_idx1[:N]]
        training_x = np.concatenate([x0, x1], axis=0)
        training_y = [0] * N + [1] * N

        models.append(svm.SVC(C=C, kernel=kernel, gamma='auto', cache_size=1024, max_iter=-1, probability=True))
        models[i].fit(training_x, training_y)

    all_models = [lsvc, model_select, models]

    print('completed training model')


    filename = 'team05_model.pkl'
    pickle.dump(all_models, open(filename, 'wb'))
    print('save complete')


if __name__ == '__main__':
    main()

