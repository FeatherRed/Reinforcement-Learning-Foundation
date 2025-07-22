'''
    Specify a starting noise scale.
    If the performance in an episode improves, decrease the noise scale.
        In our case, we take half of the scale, but set 0.0001 as the lower bound.
    If the performance in an episode drops, increase the noise scale.
        In our case, we double the scale, but set 2 as the upper bound.
'''

import torch
import gym
import matplotlib.pyplot as plt

def run_episode(env,weight):
    state = env.reset()[0]
    total_reward = 0
    is_done_1 = False
    is_done_2 = False
    while not is_done_1 and not is_done_2:
        state = torch.from_numpy(state).float()
        action = torch.argmax(torch.matmul(state,weight))
        state,reward,is_done_1,is_done_2,info = env.step(action.item())
        total_reward += reward
    return total_reward

env = gym.make('CartPole-v0',render_mode='rgb_array')
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_episode = 1000
noisy_scale = 0.01    #噪声强度
total_rewards = []
best_reward = 0
best_weight = torch.rand(n_state,n_action)

for episode in range(n_episode):
    weight = best_weight + noisy_scale * torch.rand(n_state,n_action)
    total_reward = run_episode(env,weight)
    print('Episode {}: {}'.format(episode + 1, total_reward))
    if best_reward <= total_reward:
        best_weight = weight
        best_reward = total_reward
        #decrease the noise scale
        noisy_scale = max(1e-4,noisy_scale/2)
    else:
        #increase the noise scale
        noisy_scale = min(2,noisy_scale*2)
    total_rewards.append(total_reward)

print('Average total reward over {} episode: {}'.format(n_episode, sum(total_rewards) / n_episode))



# plt.plot(total_rewards)
# plt.xlabel('Episodes')
# plt.ylabel('Reward')
# plt.show()

n_episode_eval = 1000
total_rewards_eval = []
for episode in range(n_episode_eval):
    total_reward = run_episode(env,best_weight)
    print('Episode {}: {}'.format(episode + 1, total_reward))
    total_rewards_eval.append(total_reward)

print('Average total reward over {} episode: {}'.format(n_episode, sum(total_rewards) / n_episode))