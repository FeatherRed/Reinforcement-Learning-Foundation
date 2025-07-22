import gym
import torch
from collections import defaultdict

def gen_random_policy(n_action):
    probs = torch.ones(n_action) / n_action
    def policy_function(state):
        return probs
    return policy_function

def run_episode(env,behavior_policy):
    state = env.reset()[0]
    rewards = []
    actions = []
    states = []
    is_done_1 = False
    is_done_2 = False
    while not is_done_1 and not is_done_2:
        probs = behavior_policy(state)
        action = torch.multinomial(probs,1).item()
        '''存轨迹'''
        states.append(state)
        actions.append(action)
        state,reward,is_done_1,is_done_2,info = env.step(action)
        rewards.append(reward)
    return states, actions, rewards

def mc_control_off_policy(env, gamma, n_episode, behavior_policy):
    n_action = env.action_space.n
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.empty(n_action))
    for episode in range(n_episode):
        W = defaultdict(float)
        w = 1
        states_t, actions_t, rewards_t = run_episode(env,behavior_policy)
        return_t = 0
        G = {}
        for state_t, action_t, reward_t in zip(states_t[::-1],actions_t[::-1],rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[(state_t,action_t)] = return_t
            if action_t != torch.argmax(Q[state_t]).item():
                break
            '''
                或者这么理解，从后往前更新的时候，只需要遇到第一次动作不同，然后再优化Q函数，直到轨迹中每一步的动作都与
                behavior policy采样的动作一样，这说明target policy学习到了
            '''
            w *= 1./behavior_policy(state_t)[action_t]
            #需要存W的值
            W[(state_t,action_t)] = w
        for state_action, reward in G.items():
            state, action = state_action
            if state[0] <= 21:
                G_sum[state_action] += reward * W[state_action]
                N[state_action] += 1
                Q[state][action] = G_sum[state_action] / N[state_action]
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

def mc_control_off_policy_incremental(env, gamma, n_epsiode, behavior_policy):
    n_action = env.action_space.n
    N = defaultdict(int)
    Q = defaultdict(lambda:torch.empty(n_action))
    for episode in range(n_episode):
        W = 1
        states_t, actions_t, rewards_t = run_episode(env,behavior_policy)
        return_t = 0
        for state_t, action_t, reward_t in zip(states_t[::-1],actions_t[::-1],rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            N[(state_t,action_t)] += 1
            Q[state_t][action_t] += (W / N[(state_t,action_t)]) * (return_t - Q[state_t][action_t])
            if action_t != torch.argmax(Q[state_t]).item():
                break
            W *= 1./behavior_policy(state_t)[action_t]
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

def mc_control_off_policy_weighted(env, gamma, n_episode, behavior_policy):
    n_action = env.action_space.n
    N = defaultdict(int)
    Q = defaultdict(lambda:torch.empty(n_action))
    for episode in range(n_episode):
        W = 1
        states_t, actions_t, rewards_t = run_episode(env,behavior_policy)
        return_t = 0
        for state_t, action_t, reward_t in zip(states_t[::-1],actions_t[::-1],rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            N[(state_t,action_t)] += W
            Q[state_t][action_t] += (W / N[(state_t,action_t)]) * (return_t - Q[state_t][action_t])
            if action_t != torch.argmax(Q[state_t]).item():
                break
            W *= 1./behavior_policy(state_t)[action_t]
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

def simulate_episode(env,policy):
    state = env.reset()[0]
    rewards = 0
    is_done_1 = False
    is_done_2 = False
    while not is_done_1 and not is_done_2:
        action = policy[state]
        state,reward,is_done_1,is_done_2,_ = env.step(action)
        rewards += reward
    return rewards


env = gym.make('Blackjack-v1',render_mode='rgb_array')
n_action = env.action_space.n
gamma = 1
n_episode = 500000
random_policy = gen_random_policy(n_action)
optimal_Q, optimal_policy = mc_control_off_policy_weighted(env,gamma,n_episode,random_policy)
#print(optimal_Q)
#print(optimal_policy)

n_episode = 100000
n_win_optimal = 0
n_lose_optimal = 0
for episode in range(n_episode):
    reward = simulate_episode(env,optimal_policy)
    if reward == 1:
        n_win_optimal += 1
    elif reward == -1:
        n_lose_optimal += 1
print('Winning probability under the optimal policy: {}'.format(n_win_optimal/n_episode))
print('Losing probability under the optimal policy: {}'.format(n_lose_optimal/n_episode))
'''
Winning probability under the optimal policy: 0.41415
Losing probability under the optimal policy: 0.50107
'''