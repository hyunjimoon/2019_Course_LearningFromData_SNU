import pickle
import DataGenerator
from sklearn.metrics import accuracy_score, recall_score, f1_score


def main():

    test_day = ['2019-04-13', '2019-04-14', '2019-04-20', '2019-04-21']
    test_x, test_y = DataGenerator.get_data(test_day, is_training=False)

    # TODO: fix pickle file name
    filename = 'team00_model.pkl'
    model = pickle.load(open(filename, 'rb'))
    print('load complete')
    print(model.get_params())

    # ================================ predict result ========================================
    pred_y = model.predict(test_x)

    print('accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    print('recall: {}'.format(recall_score(test_y, pred_y)))
    print('f1-score: {}'.format(f1_score(test_y, pred_y)))

if __name__ == '__main__':
    main()