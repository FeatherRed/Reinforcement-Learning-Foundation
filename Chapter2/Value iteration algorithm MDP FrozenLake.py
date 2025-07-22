import torch
import gym
import matplotlib.pyplot as plt

def run_episode(env,policy):
    state = env.reset()[0]
    total_reward = 0
    is_done_1 = False
    is_done_2 = False

    while not is_done_1 and not is_done_2:
        action = policy[state].item()
        state,reward,is_done_1,is_done_2,info = env.step(action)
        total_reward += reward
    return total_reward

def value_iteration(env,gamma,threshold):
    env.reset()
    n_action = env.action_space.n
    n_state = env.observation_space.n
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(n_state):
            V_action = torch.zeros(n_action)
            for action in range(n_action):
                for prob,new_state,reward,_ in env.env.P[state][action]:
                  V_action[action] += prob * (reward + gamma * V[new_state])
            V_temp[state] = torch.max(V_action)
        delta_max = torch.max(torch.abs(V-V_temp))
        if delta_max <= threshold:
            break
        V = V_temp.clone()
    return V

def extract_optimal_policy(env,V_optimal,gamma):
    env.reset()
    n_action = env.action_space.n
    n_state = env.observation_space.n
    policy = torch.zeros(n_state)

    for state in range(n_state):
        policy_action = torch.zeros(n_action)
        for action in range(n_action):
            for prob,new_state,reward,_ in env.env.P[state][action]:
                policy_action[action] += prob * (reward + gamma * V_optimal[new_state])
        policy[state] = torch.argmax(policy_action)
    return policy

env = gym.make('FrozenLake-v1',render_mode='rgb_array')
n_action = env.action_space.n
n_state = env.observation_space.n
gamma = 0.99     #learning rate
threshold = 1e-4

V_optimal = value_iteration(env,gamma,threshold)
optimal_policy = extract_optimal_policy(env,V_optimal,gamma)
print('Optimal values:\n{}'.format(V_optimal))
print('Optimal policy:\n{}'.format(optimal_policy))
'''
Optimal values:
tensor([0.5403, 0.4966, 0.4680, 0.4540, 0.5569, 0.0000, 0.3571, 0.0000, 0.5905,
        0.6421, 0.6143, 0.0000, 0.0000, 0.7410, 0.8625, 0.0000])
Optimal policy:
tensor([0., 3., 3., 3., 0., 0., 0., 0., 3., 1., 0., 0., 0., 2., 1., 0.])
'''


# ##使用求解的Policy 测试
# n_episode = 1000
# total_rewards = []
# for episode in range(n_episode):
#     total_reward = run_episode(env,optimal_policy)
#     print('Episode {}: {}'.format(episode + 1, total_reward))
#     total_rewards.append(total_reward)
#
# print('Average total reward under the optimal policy: {}'.format(sum(total_rewards) / n_episode))

# plt.plot(total_rewards)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.show()
#
# gammas = [0,0.2,0.4,0.6,0.8,.99,1.]
# avg_reward_gamma = []
# for gamma in gammas:
#     V_optimal = value_iteration(env,gamma,threshold)
#     optimal_policy = extract_optimal_policy(env,V_optimal,gamma)
#     total_rewards = []
#     for episode in range(n_episode):
#         total_reward = run_episode(env,optimal_policy)
#         total_rewards.append(total_reward)
#     avg_reward_gamma.append(sum(total_rewards) / n_episode)
#
# plt.plot(gammas,avg_reward_gamma)
# plt.title('Success rate vs discount factor')
# plt.xlabel('Discount factor')
# plt.ylabel('Average success rate')
# plt.show()


