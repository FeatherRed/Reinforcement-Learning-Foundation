'''
    According to the following wiki page, https:/​/​github.​com/​openai/​gym/​wiki/​CartPole-​v0,
    solved means the average reward over 100 consecutive episodes is no less than 195.
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
        #decrease the noise scale
        noisy_scale = max(1e-4,noisy_scale/2)
    else:
        #increase the noise scale
        noisy_scale = min(2,noisy_scale*2)
    total_rewards.append(total_reward)
    if episode >= 99 and sum(total_rewards[-100:]) >= 19500:
        break

#print('Average total reward over {} episode: {}'.format(n_episode, sum(total_rewards) / n_episode))
# plt.plot(total_rewards)
# plt.xlabel('Episodes')
# plt.ylabel('Reward')
# plt.show()

