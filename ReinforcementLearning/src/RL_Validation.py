
from decision_ql import QLearningDecisionPolicy
import DataGenerator as DataGenerator
import numpy as np
import simulation as simulation
import tensorflow as tf
import pandas as pd
import RL_Test as RL_Test
from datetime import timedelta



def k_validation(budget, num_stocks, start_date, end_date, time_step, k, company_list, num_epoch):
    p_result = []
    for p in range(k):
        print(p)
        tf.reset_default_graph()
        last_date = pd.to_datetime(end_date) - timedelta(days=time_step * p)
        portfolios = list()

        print('#' * 100)
        print('COMPANY : {}'.format(company_list))
        open_prices, close_prices, features = DataGenerator.make_features(start_date, last_date, company_list,
                                                                                   is_training=True)

        noise = np.random.random() * 0.1
        policy = QLearningDecisionPolicy(epsilon, epsilon_decay, gamma, lr, actions,
                                         len(features[0]) + state_dict[dim_state],
                                         'model')
        simulation.run_simulations(company_list, policy, budget * (1 - noise), num_stocks, dim_state,
                                   open_prices, close_prices, features, num_epoch)
        open_prices, close_prices, features = DataGenerator.make_features(start_date, last_date, company_list,
                                                                                   is_training=False)
        value = RL_Test.eval(
            policy, budget, num_stocks, dim_state, open_prices, close_prices, features)

        policy = QLearningDecisionPolicy(epsilon, epsilon_decay, gamma, lr, actions,
                                            len(features[0]) + state_dict[dim_state],
                                            "LFD_project4_team05-d{}-e{}".format(dim_state, num_epoch))

        value_best = RL_Test.eval(
            policy, budget, num_stocks, dim_state, open_prices, close_prices, features)

        portfolios.append([value, value_best])
        print(company_list)
        print(value, value_best)
        tf.reset_default_graph()

        print(p)
        print(portfolios)
        p_result.append(portfolios)
        print('#############################')

    return p_result


if __name__ == '__main__':

    ################################################
    # company_list = ['c', 'lgchem', 'lgnh', 's', 's2', 'sk']
    company_list = ['c', 'lgchem', 's2', 'sk']

    epsilon = 1.0
    epsilon_decay = 0.999
    inflation = 0.0004
    gamma = 1 / (1 + inflation)
    lr = 0.001
    num_epoch = 500

    dim_state = 0 # state : features
    # dim_state = 1 # state : features + open_price

    ### 밑에 두 개는 multi-step이 아닐 때만 가능할 듯
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

    time_step = 14
    k = 3

    # budget_ratio = {'c': 1, 'lgchem': 1, 'lgnh': 1, 's': 1, 's2': 1, 'sk': 1 }
    # total_ratio = 0
    # for c in company_list:
    #     total_ratio += budget_ratio[c]
    # budget_dict = { c : budget * budget_ratio[c] / total_ratio for c in company_list }


    k_final = k_validation(budget, num_stocks, start, end, time_step, k, company_list, num_epoch)

    for p in k_final:
        print('final')
        print("Final portfolio: {} won".format(p))

