from decision_ql import QLearningDecisionPolicy
import DataGenerator
import simulation


if __name__ == '__main__':

    open_prices, close_prices, features = DataGenerator.make_features('2010-01-01', '2019-05-08', is_training=True)

    # TODO: define action
    actions = ["Buy", "Sell", "Hold"]

    policy = QLearningDecisionPolicy(actions, len(features[0]) + 2, "model")

    budget = 100000000.0
    num_stocks = 0
    num_epoch = 10
    simulation.run_simulations(policy, budget, num_stocks, open_prices, close_prices, features, num_epoch)
    ######## run_simulations는 어디서 온 함수?

    # TODO: fix checkpoint directory name
    policy.save_model("LFD_project4_team00", num_epoch)
