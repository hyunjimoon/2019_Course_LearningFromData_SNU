import numpy as np


def do_action(action, budget, num_stocks, stock_price):
    # TODO: define action's operation
    if action == "Buy" and budget >= stock_price:
        budget -= stock_price
        num_stocks += 1
    elif action == "Sell" and num_stocks > 0:
        budget += stock_price
        num_stocks -= 1
    else:
        action = "Hold"

    return budget, num_stocks, action


def run_simulation(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features):
    budget = initial_budget
    num_stocks = initial_num_stocks
    stock_price = 0

    for i in range(len(open_prices)-1):
        # TODO: define state
        current_state = np.asmatrix(np.hstack((features[i], budget, num_stocks)))

        # calculate current portfolio value
        stock_price = float(open_prices[i])
        current_portfolio = budget + num_stocks * stock_price

        # select action
        action = policy.select_action(current_state, i)

        # update portfolio values based on action
        budget, num_stocks, action = do_action(action, budget, num_stocks, stock_price)

        # calculate new portofolio after taking action
        stock_price = float(close_prices[i])
        new_portfolio = budget + num_stocks * stock_price

        # calculate reward from taking an action at a state
        # TODO: define reward
        reward = new_portfolio - current_portfolio

        # TODO: define state
        next_state = np.asmatrix(np.hstack((features[i+1], budget, num_stocks)))

        # update the policy after experiencing a new action
        policy.update_q(current_state, action, reward, next_state)

    # compute final portfolio worth
    portfolio = budget + num_stocks * stock_price

    print('budget: {}, shares: {}, stock price: {} =>  portfolio: {}'.format(budget, num_stocks, stock_price, portfolio))
    return portfolio


def run_simulations(policy, budget, num_stocks, open_prices, close_prices, features, num_epoch):
    final_portofolios = list()

    for i in range(num_epoch):
        print("simuration no.{}".format(i))
        final_portofolio = run_simulation(policy, budget, num_stocks, open_prices, close_prices, features)
        final_portofolios.append(final_portofolio)

    print(final_portofolios[-1])
