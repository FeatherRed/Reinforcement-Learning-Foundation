import gym
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

def gen_epsilon_greedy_policy(n_action, epsilon):
    def policy_function(state, Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function

def sarsa(env, gamma, n_episode, alpha):
    n_action = env.action_space.n
    n_state = env.observation_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in range(n_episode):
        state = env.reset()[0]
        is_done_1 = False
        is_done_2 = False
        action = epsilon_greedy_policy(state, Q)
        while not is_done_1 and not is_done_2:
            next_state, reward, is_done_1, is_done_2, info = env.step(action)
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            next_action = epsilon_greedy_policy(next_state, Q)
            td_delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            Q[state][action] += alpha * td_delta

            state = next_state
            action = next_action
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy


env = gym.make("Taxi-v3",render_mode="rgb_array")
alpha_options = [0.4, 0.5, 0.6]
epsilon_options = [0.1, 0.03, 0.01]
gamma = 1
n_episode = 500
n_state = env.observation_space.n
n_action = env.action_space.n
for alpha in alpha_options:
    for epsilon in epsilon_options:
        length_episode = [0] * n_episode
        total_reward_episode = [0] * n_episode
        epsilon_greedy_policy = gen_epsilon_greedy_policy(n_action, epsilon)
        sarsa(env, gamma, n_episode, alpha)
        reward_per_step = [reward/float(step) for reward, step in zip(total_reward_episode, length_episode)]
        print('alpha: {}, epsilon: {}'.format(alpha, epsilon))
        print('Average reward over {} episodes: {}'.format(n_episode, sum(total_reward_episode) / n_episode))
        print('Average length over {} episodes: {}'.format(n_episode, sum(length_episode) / n_episode))
        print('Average reward per step over {} episodes: {}\n'.format(n_episode, sum(reward_per_step) / n_episode))

