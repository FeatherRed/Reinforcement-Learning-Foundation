'''
In the hill-climbing algorithm, we also start with a randomly chosen weight.
But here, for every episode, we add some noise to the weight.
If the total reward improves, we update the weight with the new one; otherwise, we keep the old weight.
In this approach, the weight is gradually improved as we progress through the episodes, instead of jumping around in each episode.
'''

import torch
import gym

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
    total_rewards.append(total_reward)

print('Average total reward over {} episode: {}'.format(n_episode, sum(total_rewards) / n_episode))
