from tensorflow.python.platform import gfile
from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
from collections import deque
import math


MEMORY_SIZE = 100000 # EPISODE_SIZE * 100
BATCH_SIZE = int(MEMORY_SIZE * 0.2)

class QLearningDecisionPolicy:
    def __init__(self, epsilon, epsilon_decay, gamma, lr, actions, input_dim, model_dir, graph=None, reuse=False):
        # TODO: tuning hyperparameters
        self.epsilon = epsilon  # select action function hyperparameter
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma # q functions hyperparameter
        self.lr = lr # neural network hyperparmeter

        self.memory_size = MEMORY_SIZE
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.actions = actions # buy / hold / sell
        self.action_dict = {actions[i] : i for i in range(len(actions))}
        self.multi_step = 3
        # self.multi_step = 1

        # Neural Network
        self.num_atoms = 1001  # 51 for C51
        # self.v_max = 4 * 10 ** 6  # Max possible score for Defend the center is 26 - 0.1*26 = 23.4
        # self.v_min = -4 * 10 ** 6  # -0.1*26 - 1 = -3.6
        self.v_max = 2 * 10 ** 7  # Max possible score for Defend the center is 26 - 0.1*26 = 23.4
        self.v_min = -2 * 10 ** 7  # -0.1*26 - 1 = -3.6

        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        self.input_dim = input_dim
        self.output_dim = len(actions)

        if graph == None:
            self.x = tf.placeholder(tf.float32, [None, self.input_dim])
            self.y = tf.placeholder(tf.float32, [None, self.output_dim, self.num_atoms])
            self.q = self.build_network('main')
            self.target_q = self.build_network('target')
            self.loss = tf.losses.huber_loss(self.q, self.y)
            self.sess = tf.Session()
            with tf.variable_scope('Adam', reuse=tf.AUTO_REUSE):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            # restore model
            self.saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("load model: %s" % ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        else:
            with graph.as_default():
                self.x = tf.placeholder(tf.float32, [None, self.input_dim])
                self.y = tf.placeholder(tf.float32, [None, self.output_dim, self.num_atoms])
                self.q = self.build_network('main')
                self.target_q = self.build_network('target')
                self.loss = tf.losses.huber_loss(self.q, self.y)
                with tf.variable_scope('Adam', reuse=reuse):
                    if reuse:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

                init_op = tf.global_variables_initializer()

            self.sess = tf.Session(graph=graph)

            with self.sess.graph.as_default():
                self.sess.run(init_op)
                # restore model
                self.saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(model_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("load model: %s" % ckpt.model_checkpoint_path)
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)


    def build_network(self, name):
        ### DistriDistributional Dualing DDQN
        with tf.variable_scope(name):
            # DQN = self.dist_mlp(hiddens=[128, 64, 32, 8], inpt=self.x, num_actions=self.output_dim,
            DQN = self.dist_mlp(hiddens=[128, 64, 32], inpt=self.x, num_actions=self.output_dim,
                                n_atoms=self.num_atoms, scope=name, layer_norm=True)
            return DQN

    def dist_mlp(self, hiddens, inpt, num_actions, n_atoms, scope, reuse=tf.AUTO_REUSE, layer_norm=False):
        ### Distributional Dualing DDQN
        with tf.variable_scope(scope, reuse=reuse):
            out = inpt
            with tf.variable_scope("DNN"):
                for hidden in hiddens:
                    out = tf.layers.dense(out, units=hidden, activation=tf.nn.relu6)
                    if layer_norm:
                        out = tf.layers.batch_normalization(out)
                    out = tf.nn.leaky_relu(out)
            with tf.variable_scope("action_value"):
                action_scores = tf.layers.dense(out, units=num_actions * n_atoms, activation=tf.nn.relu6)
                action_scores = tf.reshape(action_scores, shape=[-1, num_actions, n_atoms])
            with tf.variable_scope("state_value"):
                state_score = tf.layers.dense(out, units=1, activation=tf.nn.relu6)

            q_out = tf.add(tf.tile(tf.expand_dims(state_score, -1), [1, num_actions, n_atoms]), action_scores)

        return q_out

    def select_action(self, current_state, is_training=True):
        if random.random() > self.epsilon or not is_training:
            ### Distributional Dualing DDQN
            action_p_vals = np.vstack(self.sess.run(self.q, feed_dict={self.x: [current_state]}))
            action_q_vals = np.sum(np.multiply(action_p_vals, np.array(self.z)), axis=1)
            action_idx = np.argwhere(action_q_vals == np.amax(action_q_vals))
            np.random.shuffle(action_idx)
            action = self.actions[action_idx[0][0]]

            # if not is_training:
            #     print(action_q_vals)

        else:
            if random.random() < 1/3:
                action = 'nothing'
            else:
                action = self.actions[random.randint(0, len(self.actions) - 2)]


        return action


    def update_q(self):
        batch_size = min(len(self.memory), BATCH_SIZE)
        current_state, action, reward, next_state, done = self.sample_experience(batch_size=batch_size)
        m_prob = np.zeros((batch_size, self.output_dim, self.num_atoms))
        ### Distributional Dualing DDQN
        target_p_vals = np.vstack(self.sess.run(self.target_q, feed_dict={self.x: next_state}))
        target_q_vals = np.sum(np.multiply(target_p_vals, np.array(self.z)), axis=1)  # length (num_atoms x num_actions)
        target_q_vals = target_q_vals.reshape((batch_size, self.output_dim), order='F')
        optimal_target_idxs = np.argmax(target_q_vals, axis=1)

        print(target_q_vals[0])
        print('%.3f ~ %.3f ~ %.3f'%(np.quantile(reward, 0.25), np.quantile(reward, 0.5), np.quantile(reward, 0.75)))

        for i in range(batch_size):
            if done:
                Tz = min(self.v_max, max(self.v_min, reward[i] ** self.multi_step))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[i, self.action_dict[action[i]], int(m_l)] += (m_u - bj)
                m_prob[i, self.action_dict[action[i]], int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max,
                             max(self.v_min, reward[i] ** self.multi_step + self.gamma ** self.multi_step * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[i, self.action_dict[action[i]], int(m_l)] += target_p_vals[
                                                                            optimal_target_idxs[i], i ,j] * (m_u - bj)
                    m_prob[i, self.action_dict[action[i]], int(m_u)] += target_p_vals[
                                                                            optimal_target_idxs[i], i ,j] * (bj - m_l)
        self.sess.run(self.train_op, feed_dict={self.x: current_state, self.y: m_prob})

    def update_target_q(self):
        holder = []
        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES, scope = 'target')
        for main_var, target_var in zip(main_vars, target_vars):
            holder.append(target_var.assign(main_var.value()))
        self.sess.run(holder)

    def store_experience(self, experience):
        self.memory.append(experience)
        while len(self.memory) > self.memory_size:
            self.memory.popleft()

    def sample_experience(self, batch_size = 1):
        minibatch = random.sample(self.memory, batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def save_model(self, output_dir):
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)

        checkpoint_path = output_dir + '/model'
        self.saver.save(self.sess, checkpoint_path)