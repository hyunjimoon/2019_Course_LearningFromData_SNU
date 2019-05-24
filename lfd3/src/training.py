import pystan
from sklearn import svm
import pickle
import DataGenerator


def main():

    test_day = ['2019-04-13', '2019-04-14', '2019-04-20', '2019-04-21']
    training_x, training_y = DataGenerator.get_data(test_day, is_training=True)

    # ================================ train SVM model=========================================
    # TODO: set parameters
    print('start training model')
    model = svm.SVC()
    model.fit(training_x, training_y)

    print('completed training model')

    # TODO: fix pickle file name
    filename = 'team00_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('save complete')


if __name__ == '__main__':
    main()
