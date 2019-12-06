from decision_ql import QLearningDecisionPolicy
import DataGenerator as DataGenerator
import simulation as simulation
import tensorflow as tf
from matplotlib import pyplot as plt

if __name__ == '__main__':
    ################################################
    # company_list = ['c', 'lgchem', 'lgnh', 's', 's2', 'sk']
    company_list = ['c', 'lgchem', 's2', 'sk']

    epsilon = 1.0
    epsilon_decay = 0.999
    inflation = 0.0004
    gamma = 1 / (1 + inflation)
    lr = 0.0001
    num_epoch = 10000

    dim_state = 0 # state : features
    # dim_state = 1 # state : features + open_price
    # dim_state = 2 # state : features + budget + num_stock
    # dim_state = 3  # state : features + open_price + budget + num_stock
    ################################################
    budget = 10. ** 8
    num_stocks = 0
    start, end = '2010-01-01', '2019-05-08'  # start date 짧아도 무관
    state_dict = {0: 0,
                  1: len(company_list),
                  2: len(company_list) + 1,
                  3: 2 * len(company_list) + 1}
    actions = company_list + ['nothing']
    ################################################

    print('#' * 100)
    print('COMPANY : {}\n'.format(company_list))
    open_prices, close_prices, features = DataGenerator.make_features(start, end, company_list, is_training=True)

    policy = QLearningDecisionPolicy(epsilon, epsilon_decay, gamma, lr, actions,
                                     len(features[0]) + state_dict[dim_state],
                                     'model')
    portfolios = simulation.run_simulations(company_list, policy, budget, num_stocks, dim_state,
                               open_prices, close_prices, features, num_epoch)

    plt.title('Company : {}  /  State dimension : {}'.format(company_list, dim_state))
    plt.plot(portfolios)
    plt.axhline(y=budget, xmin=0, xmax=num_epoch-1, color='r')
    plt.show()

    # policy.save_model("MULTI_{}-d{}-e{}".format(company_list, dim_state, num_epoch))

    tf.reset_default_graph()

    print('#' * 100)
