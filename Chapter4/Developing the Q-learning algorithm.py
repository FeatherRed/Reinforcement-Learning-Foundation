import gym
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
def gen_episode_greedy_policy(n_action, epsilon):
    def policy_function(state, Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs,1).item()
        '''
        torch.multionmial(input, num_samples, replacement=False, out=None) -> LongTensor
        对input 采样，次数num_samples，replacement:是否放回，默认不返回
        '''
        return action
    return policy_function

def q_learning(env, gamma, n_episode, alpha):
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
        state = env.reset()[0]
        is_done_1 = False
        is_done_2 = False
        while not is_done_1 and not is_done_2:
            action = episode_greedy_policy(state,Q)
            next_state, reward, is_done_1, is_done_2, info = env.step(action)
            # td = 1 一步学习
            td_delta = reward + gamma * torch.max(Q[next_state]) - Q[state][action]
            Q[state][action] += alpha * td_delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            state = next_state
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

env = gym.make("CliffWalking-v0",render_mode="rgb_array")
gamma = 1
n_episode = 500
alpha = 0.4
epsilon = 0.1
length_episode = [0] * n_episode
total_reward_episode = [0] * n_episode
episode_greedy_policy = gen_episode_greedy_policy(env.action_space.n, epsilon)
optimal_Q, optimal_policy = q_learning(env,gamma,n_episode,alpha)
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
 {36: 0, 24: 1, 12: 0, 0: 2, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 1, 19: 2, 18: 1, 17: 1, 16: 1, 15: 2, 14: 2, 13: 0, 30: 1, 31: 1, 20: 1, 9: 2, 10: 2, 11: 2, 23: 2, 22: 2, 35: 2, 21: 2, 32: 1, 29: 1, 34: 1, 33: 1, 28: 1, 27: 1, 25: 1, 26: 1, 47: 0}
'''
'''Env = gym.make("CliffWalking-v0",render_mode="human")
state = Env.reset()[0]
is_done_1 = False
is_done_2 = False
while not is_done_1 and not is_done_2:
    action = optimal_policy[state]
    state, reward, is_done_1, is_done_2, info = Env.step(action)
'''
