from decision_ql import QLearningDecisionPolicy
import DataGenerator as DataGenerator
import numpy as np
from simulation import do_action
import tensorflow as tf

def do_action_ensemble(action_list, prev_action, action, budget, num_stocks_list, stock_price_list):
    # TODO: define action's operation
    # print(action_list, prev_action, action, budget, num_stocks_list, stock_price_list)
    num_stocks_trade_list = [0] * (len(action_list)-1)
    trade_budget = 0
    for i, a in enumerate(action_list[:-1]):
        if prev_action != a and action == a: # 사기. budget에 변화가 오면 안 되므로 사는 행위를 파는 행위보다 먼저 수행
            # print(action, 'buy')
            buy_num = max(budget // 4 // stock_price_list[i], 1)
            num_stocks_trade_list[i] = buy_num
            num_stocks_list[i] = buy_num
            budget -= stock_price_list[i] * buy_num

    for i, a in enumerate(action_list[:-1]):
        if prev_action == a:
            if action == a: # a를 냅두기
                # print(prev_action, 'nothing')
                continue
            else: # a를 팔기
                # print(prev_action, 'sell')
                buy_num = - num_stocks_list[i]
                num_stocks_trade_list[i] = buy_num
                num_stocks_list[i] = 0
                budget -= stock_price_list[i] * buy_num

    return budget, num_stocks_list, num_stocks_trade_list, action


def eval(graph, policy, initial_budgets, initial_num_stocks, dim_state, open_prices, close_prices, features):

    budgets = initial_budgets
    num_stocks_lists = dict()
    num_stocks_trade_lists = dict()
    actions = dict()

    for i in policy:
        if initial_num_stocks == 0:
            num_stocks_lists[i] = [0] * (len(policy[i].actions) - 1)
        elif type(initial_num_stocks) == list:
            num_stocks_lists[i] = initial_num_stocks
        else:
            raise print('initial num stock wrong')
        actions[i] = 'nothing'

    budget = sum(budgets)
    num_stocks_aggregated = np.sum(list(num_stocks_lists.values()), axis=0)

    for t in range(len(open_prices)):
        print('Day {}'.format(t))
        trade_budgets = 0
        for i in policy:
            with graph[i].as_default():
                prev_actions = actions.copy()
                if dim_state == 0:
                    current_state = np.concatenate([features[t]])
                actions[i] = policy[i].select_action(current_state, is_training=False)
                budgets[i], num_stocks_lists[i], num_stocks_trade_lists[i], actions[i] = \
                    do_action_ensemble(policy[i].actions, prev_actions[i], actions[i], budgets[i], num_stocks_lists[i], open_prices[t])

        budget = sum(budgets)
        num_stocks_aggregated = np.sum(list(num_stocks_lists.values()), axis=0)

        print('action {} / budget {} / shares {}'.format(actions, budget, num_stocks_aggregated))
        print('portfolio {}'.format(budget + sum(num_stocks_aggregated * close_prices[t])))

    portfolio = budget + sum(num_stocks_aggregated * close_prices[-1])

    print('Finally, you have')
    print('budget: %.3f won' % budget)
    print('Share value : %.3f won {},{}'.format(close_prices[-1], num_stocks_aggregated))
    print('Portfolio: %.3f won' % portfolio)
    print()

    return portfolio


if __name__ == '__main__':

    ################################################
    company_list = ['c', 'lgchem', 's2', 'sk']
    num_epochs = [147, 550]
    port_ratio = [0.2908196507400389, 0.7091803492599611]
    # num_epochs = [147]
    # num_epochs = [550]

    dim_state = 0 # state : features
    ################################################
    budget = 10. ** 8
    budgets = [budget * r for r in port_ratio]
    num_stocks = 0
    start, end = '2010-01-01', '2019-05-08'  # start date 짧아도 무관
    state_dict = {0: 0,
                  1: len(company_list),
                  2: len(company_list) + 1,
                  3: 2 * len(company_list) + 1}
    actions = company_list + ['nothing']
    ################################################

    open_prices, close_prices, features = DataGenerator.make_features(start, end, company_list,
                                                                      is_training=False)


    print('#' * 100)
    print('Company {} / Epoch {}'.format(company_list, num_epochs))

    policy = dict()
    graph = [tf.Graph()] * 2

    for i, num_epoch in enumerate(num_epochs):
        policy[i] = QLearningDecisionPolicy(0, 0, 1, 0, actions,
                                    len(features[0]) + state_dict[dim_state],
                                    "LFD_team05_project4-d{}-e{}".format(dim_state, num_epoch), graph[i], bool(i))



    final_portfolio = eval(graph, policy, budgets, num_stocks, dim_state, open_prices, close_prices, features)

    tf.reset_default_graph()

    print('#' * 100)

    print("Final portfolio: %.3f won" % final_portfolio)



    ##################################################################