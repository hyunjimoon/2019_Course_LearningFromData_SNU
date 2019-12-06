from pandas.plotting import register_matplotlib_converters
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
register_matplotlib_converters()

EPISODE_SIZE = 10
START_EPOCH = 20
START_EPOCH = 100
DECAY_EPOCH = 100
DECAY_EPOCH = 1000
Q_UPDATE_EPISODE_INTERVAL = 20 # epoch 단위로 학습하고 한 epoch에 약 100*0.2개의 에피소드가 존재하므로 사실상 전부 한 번씩 학습하는 형태
TARGET_UPDATE_EPISODE_INTERVAL = Q_UPDATE_EPISODE_INTERVAL * 5
PRINT_EPOCH = 100

def do_action(action_list, prev_action, action, budget, num_stocks_list, stock_price_list):
    # TODO: define action's operation
    num_stocks_trade_list = [0] * (len(action_list)-1)
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


def run_simulation(c, policy, initial_budget, initial_num_stocks,  dim_state, open_prices, close_prices, features, epoch):
    count = 0
    portfolios = []
    action_count = [0] * len(policy.actions)
    action_seq = list()
    EPISODE_SIZE = len(open_prices) - policy.multi_step - 1
    # for i in range(0, len(open_prices) - EPISODE_SIZE - 1, EPISODE_SIZE):
    for i in range(0, len(open_prices) - EPISODE_SIZE - policy.multi_step, EPISODE_SIZE // 2):
        budget = initial_budget
        if initial_num_stocks == 0:
            num_stocks_list = [0] * (len(policy.actions) - 1)
        elif type(initial_num_stocks) == list:
            num_stocks_list = initial_num_stocks
        else:
            raise print('initial num stock wrong')
        action = 'nothing'

        for t in range(i, i + EPISODE_SIZE):
            prev_action = action
            count += 1
            ##### TODO: define current state
            if dim_state == 0:
                current_state = np.concatenate([features[t]])
            elif dim_state == 1:
                current_state = np.concatenate([features[t], open_prices[t]])
                # current_state = np.concatenate([features[t], [budget]])
            elif dim_state == 2:
                current_state = np.concatenate([features[t], [budget], num_stocks_list])
            elif dim_state == 3:
                current_state = np.concatenate([features[t], open_prices[t], [budget], num_stocks_list])

            ##### TODO: select action & define reward
            action = policy.select_action(current_state, True)
            action_seq.append(action)
            for i,a in enumerate(policy.actions):
                if action == a: action_count[i] += 1

            budget, num_stocks_list, num_stocks_trade_list, action = \
                                do_action(policy.actions, prev_action, action, budget, num_stocks_list, open_prices[t])

            reward = sum(num_stocks_list * (close_prices[t] - open_prices[t]))
            reward -= sum(num_stocks_trade_list * open_prices[t])  # num_stock_trade는 산 개수니까 리워드에서 빼야 함
            # print(action, reward)

            ##### TODO: define next state
            if dim_state == 0:
                next_state = np.concatenate([features[t + policy.multi_step]])
            if dim_state == 1:
                next_state = np.concatenate([features[t + policy.multi_step],
                                             open_prices[t + policy.multi_step]])
            if dim_state == 2:
                next_state = np.concatenate([features[t + policy.multi_step], [budget], num_stocks_list])
            elif dim_state == 3:
                next_state = np.concatenate([features[t + policy.multi_step],
                                             open_prices[t + policy.multi_step], [budget], num_stocks_list])

            done = False

            experience = (current_state, action, reward, next_state, done)
            policy.store_experience(experience)

            portfolio = budget + sum(num_stocks_list * close_prices[t])
            portfolios.append(portfolio)

        # portfolio = budget + sum(num_stocks_list * close_prices[t])
        # portfolios.append(portfolio)

    if epoch > START_EPOCH and (epoch + 1) % Q_UPDATE_EPISODE_INTERVAL == 0:
        policy.update_q()
    if epoch > START_EPOCH and (epoch + 1) % TARGET_UPDATE_EPISODE_INTERVAL == 0:
        policy.update_target_q()

    port_ratio = [portfolios[i] / portfolios[i-10] - 1.0 for i in range(10, len(portfolios))]
    # mean = float(np.mean(portfolios))
    # std = float(np.std(portfolios))
    # print('portfolios: %.3fe+7 ~ %.3fe+7\n' % (mean/10**7, std/10**7))
    # return [mean-std, median, mean+std]
    return [np.quantile(portfolios, 0.1), np.median(portfolios), np.quantile(portfolios, 0.9)],\
           [np.quantile(port_ratio, 0.1), np.median(port_ratio), np.quantile(port_ratio, 0.9)], \
           action_count, action_seq


def run_simulations(c, policy, budget, num_stocks, dim_state, open_prices, close_prices, features, num_epoch):
    final_portfolios = list()
    portfolios_ratio = list()
    best_port = 0
    for epoch in range(num_epoch):
        noise = np.random.random() * 0.1
        final_portfolio, final_portfolio_ratio, action_count, action_seq \
            = run_simulation(c, policy, budget * (1 - noise), num_stocks, dim_state, open_prices, close_prices, features, epoch)
        final_portfolios.append([i / (1 - noise) for i in final_portfolio])
        portfolios_ratio.append(final_portfolio_ratio)

        if best_port < final_portfolios[-1][1]:
            best_port = final_portfolios[-1][1]
            policy.save_model("LFD_team05_project4-d{}-e{}".format(dim_state, epoch))

        if epoch > DECAY_EPOCH:
            policy.epsilon = max(policy.epsilon_min, policy.epsilon * policy.epsilon_decay)

        if (epoch + 1) % 100 == 0:
        # if (epoch + 1) % PRINT_EPOCH == 0:
            print("-------- simulation {}    epsilon %.3f --------".format(epoch + 1) % (policy.epsilon))
            print('actions : {}'.format(action_count))

        if (epoch + 1) % PRINT_EPOCH == 0 and (epoch + 1) > DECAY_EPOCH:

            plt.title('Company {} / Epoch {}'.format(c, epoch + 1))
            plt.plot(final_portfolios)
            plt.axhline(y=budget, xmin=0, xmax=num_epoch - 1, color='r')
            plt.show()

            # plt.title('Company {} / Epoch {}'.format(c, epoch + 1))
            # plt.plot(portfolios_ratio)
            # plt.axhline(y=0.0, xmin=0, xmax=num_epoch - 1, color='r')
            # plt.show()

            action_seq = np.asarray(action_seq)
            plt.figure(figsize=(10, 10))
            plt.title('Company {} / Epoch {}'.format(c, epoch + 1))
            plt.plot(open_prices[0: len(action_seq)], 'grey')
            action_seq2 = np.concatenate([['nothing'], action_seq[:-1]])
            for i, a in enumerate(policy.actions[:-1]):
                plt.plot(pd.DataFrame(open_prices[: len(action_seq), i])[action_seq2 == a], 'ro', markersize=2) # sell
                plt.plot(pd.DataFrame(open_prices[: len(action_seq), i])[action_seq == a], 'bo', markersize=2)  # buy
            plt.show()

    return final_portfolios


