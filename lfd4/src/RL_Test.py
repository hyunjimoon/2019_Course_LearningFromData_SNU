from decision_ql import QLearningDecisionPolicy
import DataGenerator
import numpy as np
from simulation import do_action


def test(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features):

    budget = initial_budget
    num_stocks = initial_num_stocks

    for i in range(len(open_prices)):
        current_state = np.asmatrix(np.hstack((features[i], budget, num_stocks)))
        action = policy.select_action(current_state, is_training=False)
        stock_price = float(open_prices[i])

        budget, num_stocks, action = do_action(action, budget, num_stocks, stock_price)

    portfolio = budget + num_stocks * close_prices[-1]

    print('Finally, you have')
    print('budget: %f won' % budget)
    print('Shares: %i' % num_stocks)
    print('Share value: %f won' % close_prices[-1])
    print()

    return portfolio


if __name__ == '__main__':

    open_prices, close_prices, features = DataGenerator.make_features('2010-01-01', '2019-05-08', is_training=False)

    # TODO: define action
    actions = ["Buy", "Sell", "Hold"]

    budget = 100000000.0
    num_stocks = 0

    # TODO: fix checkpoint directory name
    policy = QLearningDecisionPolicy(actions, len(features[0]) + 2, "LFD_project4_team00") # 수중돈 주식개수 추가가 +2
    final_portfolio = test(policy, budget, num_stocks, open_prices, close_prices, features)

    print("Final portfolio: %f won" % final_portfolio)

