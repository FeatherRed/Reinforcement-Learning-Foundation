import gym
import torch
from collections import deque
import random
import copy
from torch.autograd import Variable
from Double_DQN import DQN

def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function

def q_learning(env, estimator, n_episode, replay_size, target_update = 10, gamma = 1.0, epsilon = 0.1, epsilon_decay = 0.99):
    n_action = env.action_space.n
    for episode in range(n_episode):
        if episode % target_update == 0:
            estimator.copy_target()
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state = env.reset()[0]
        is_done_1 = False
        is_done_2 = False
        while not is_done_1 and not is_done_2:
            action = policy(state)
            next_state, reward, is_done_1, is_done_2, info = env.step(action)
            total_reward_episode[episode] += reward
            memory.append((state, action, next_state, reward, is_done_1, is_done_2))

            if is_done_1 or is_done_2:
                break

            estimator.replay(memory, replay_size, gamma)
            state = next_state
        epsilon = max(epsilon_decay * epsilon, 0.01)


env = gym.make("CartPole-v1", render_mode = 'rgb_array')
n_action = env.action_space.n
n_state = env.observation_space.shape[0]
n_episode = 600
last_episode = 200

n_hidden_options = [30, 40]
lr_options = [0.001, 0.003]
replay_size_options = [20, 25]
target_update_options = [30, 35]

for n_hidden in n_hidden_options:
    for lr in lr_options:
        for replay_size in replay_size_options:
            for target_update in target_update_options:
                random.seed(1)
                torch.manual_seed(1)
                dqn = DQN(n_state, n_action, n_hidden, lr)
                memory = deque(maxlen = 10000)
                total_reward_episode = [0] * n_episode
                q_learning(env, dqn, n_episode, replay_size, target_update, gamma = 0.9, epsilon = 1)
                print(n_hidden, lr, replay_size, target_update, sum(total_reward_episode[-last_episode:]) / last_episode)

'''
30 0.001 20 30 219.33
30 0.001 20 35 245.88
30 0.001 25 30 160.77
30 0.001 25 35 148.31
30 0.003 20 30 217.16
30 0.003 20 35 159.17
30 0.003 25 30 173.98
30 0.003 25 35 123.46
40 0.001 20 30 151.475
40 0.001 20 35 203.575
40 0.001 25 30 166.895
40 0.001 25 35 137.85
40 0.003 20 30 110.025
40 0.003 20 35 212.31
40 0.003 25 30 385.535
40 0.003 25 35 146.635
'''