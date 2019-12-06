import pickle
import pandas as pd
import DataGenerator
from sklearn.metrics import accuracy_score, recall_score, f1_score
#print('The scikit-learn version is {}.'.format(sklearn.__version__))

def main():
    var_list = ['lane', 'F', 'M', 'G', 'K', 'age', 'weight', 'price', 'horse_money', 'rating',
                'horse_1yr_win', 'horse_total_win', 'jockey_1yr_win', 'jockey_total_win',
                'j_rank', 'h_rank']

    test_day = ['2019-04-13', '2019-04-14', '2019-04-20', '2019-04-21']
    vote_num = 4
    test_x, test_y, test_d = DataGenerator.get_data(test_day, var_list, is_training=False)

    filename = 'team05_model.pkl'
    lsvc, model_select, models = pickle.load(open(filename, 'rb'))
    print('load complete\n')

    # ================================ predict result ========================================

    test_x = model_select.transform(test_x)
    pred_ys = list()
    pred_yprobs = list()

    for i in range(len(models)):
        pred_yprobs.append(models[i].predict_proba(test_x)[:,1])
        pred_ys.append(models[i].predict(test_x))

    vote_y = [sum(x) for x in zip(*pred_yprobs)]
    test_d['real_y'] = test_y
    test_d['vote_y'] = vote_y
    test_d['pred_y'] = 0

    third = test_d.sort_values('vote_y').groupby(['date','race_num'], as_index=False).nth(-1 * vote_num)

    for idx in test_d.index:
        d, r = test_d.loc[idx, ['date','race_num']]
        threshold = third[(third.date == d) & (third.race_num == r)]['vote_y'].values[0]
        if test_d.loc[idx, 'vote_y'] >= threshold:
            test_d.loc[idx, 'pred_y'] = 1
    pred_y = test_d.pred_y


    print('accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    print('recall: {}'.format(recall_score(test_y, pred_y)))
    print('f1-score: {}\n'.format(f1_score(test_y, pred_y)))



if __name__ == '__main__':
    main()