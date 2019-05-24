from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import random


class QLearningDecisionPolicy:
    def __init__(self, actions, input_dim, model_dir):
        # TODO: tuning hyperparameters
        # select action function hyperparameter
        self.epsilon = 0.5
        # q functions hyperparameter
        self.gamma = 0.01
        # neural network hyperparmeter
        self.lr = 0.01

        self.actions = actions
        output_dim = len(actions)

        # neural network input and output placeholder
        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y = tf.placeholder(tf.float32, [output_dim])

        # TODO: build your Q-network
        # 2-layer fully connected network
        fc = tf.layers.dense(self.x, 20, activation=tf.nn.relu)
        self.q = tf.layers.dense(fc, output_dim)

        # loss
        loss = tf.square(self.y - self.q)

        # train operation
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # session
        self.sess = tf.Session()

        # initalize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # saver
        self.saver = tf.train.Saver()

        # restore model
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("load model: %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path) ######## 원래 init은 return None?

    def select_action(self, current_state, is_training=True):

        if random.random() >= self.epsilon or not is_training:
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            action_idx = np.argmax(action_q_vals)
            action = self.actions[action_idx]
        else:  # randomly select action
            action = self.actions[random.randint(0, len(self.actions)-1)]

        return action

    def update_q(self, current_state, action, reward, next_state):
        # Q(s, a)
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
        # Q(s', a')
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
        # a' index
        next_action_idx = np.argmax(next_action_q_vals)
        # create target
        action_q_vals[0, self.actions.index(action)] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        # delete minibatch dimension
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        self.sess.run(self.train_op, feed_dict={self.x: current_state, self.y: action_q_vals})

    def save_model(self, output_dir, step):
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)

        checkpoint_path = output_dir + '/model'
        self.saver.save(self.sess, checkpoint_path, global_step=step)

