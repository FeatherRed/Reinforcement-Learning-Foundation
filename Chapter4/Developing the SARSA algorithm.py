import gym
import torch
import sys
import matplotlib.pyplot as plt
from libs.envs.windy_gridworld import WindyGridworldEnv
from collections import defaultdict

def gen_epsilon_greedy_policy(n_action, epsilon):
    def policy_function(state, Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function

def sarsa(env, gamma, n_episode, alpha):
    '''

    :param env:
    :param gamma:
    :param n_episode:
    :param alpha:
    :return:
    '''
    n_action = env.action_space.n
    n_state = env.observation_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in range(n_episode):
        state = env.reset()
        is_done = False
        action = epsilon_greedy_policy(state, Q)
        while not is_done:
            next_state, reward, is_done, info = env.step(action)
            next_action = epsilon_greedy_policy(next_state, Q)
            td_delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            Q[state][action] += alpha * td_delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward

            state = next_state
            action = next_action
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

env = WindyGridworldEnv()
n_episode = 500
length_episode = [0] * n_episode
total_reward_episode = [0] * n_episode
gamma = 1
alpha = 0.4
epsilon = 0.1
epsilon_greedy_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)
optimal_Q, optimal_policy = sarsa(env, gamma, n_episode, alpha)
print('The optimal policy:\n', optimal_policy)
plt.plot(length_episode)
plt.title('Episode length over time')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.show()
plt.plot(total_reward_episode)
plt.title('Episode length over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()


'''
The optimal policy:
{30: 1, 20: 0, 10: 1, 0: 2, 1: 2, 2: 1, 12: 1, 3: 1, 4: 1, 5: 1, 6: 1, 11: 1, 13: 1, 7: 1, 8: 1, 9: 2, 19: 2, 29: 2, 18: 2, 21: 1, 22: 1, 23: 0, 39: 2, 28: 1, 32: 1, 31: 1, 33: 0, 41: 1, 40: 2, 50: 1, 42: 2, 43: 1, 14: 1, 51: 1, 24: 1, 17: 1, 52: 1, 61: 0, 53: 1, 15: 1, 60: 1, 62: 1, 34: 1, 44: 1, 63: 2, 38: 1, 49: 3, 59: 2, 48: 3, 69: 3, 58: 3, 27: 0, 54: 1, 25: 1, 16: 2, 37: 0, 45: 1, 35: 1, 68: 0, 26: 1, 47: 2, 36: 0, 57: 1}
'''
'''Env = WindyGridworldEnv()
state = Env.reset()
Env.render()
is_done = False
while not is_done:
    action = optimal_policy[state]
    state, reward, is_done, info = Env.step(action)
    Env.render()'''