

import torch
import gym
import matplotlib.pyplot as plt

def run_episode(env,weight):
    state = env.reset()[0]
    grads = []
    total_reward = 0
    is_done_1 = False
    is_done_2 = False
    while not is_done_1 and not is_done_2:
        state = torch.from_numpy(state).float()
        z = torch.matmul(state,weight)  #1x4 * 4x2 = 1x2
        probs = torch.nn.Softmax()(z)   #得到概率 1x2
        ##依据概率probs 得到action
        action = int(torch.bernoulli(probs[1]).item())
        d_softmax = torch.diag(probs) - probs.view(-1,1) * probs
        d_log = d_softmax[action] / probs[action]
        grad = state.view(-1,1) * d_log
        grads.append(grad)
        state,reward,is_done_1,is_done_2,info = env.step(action)
        total_reward += reward
    return total_reward,grads

env = gym.make('CartPole-v0',render_mode='rgb_array')
n_action = env.action_space.n
n_state = env.observation_space.shape[0]

n_episode = 1000
weight = torch.rand(n_state,n_action)
total_rewards = []
learning_rate = 1e-3

for episode in range(n_episode):
    total_reward,gradients = run_episode(env,weight)
    #print('Episode {}: {}'.format(episode + 1, total_reward))
    for i,gradient in enumerate(gradients):
        weight += learning_rate * gradient * (total_reward - i)
    total_rewards.append(total_reward)

#print('Average total reward over {} episode: {}'.format(n_episode, sum(total_rewards) / n_episode))


#画图
# plt.plot(total_rewards)
# plt.xlabel('Episodes')
# plt.ylabel('Reward')
# plt.show()

n_episode_eval = 100
total_rewards_eval = []
for episode in range(n_episode_eval):
    total_reward,_ = run_episode(env,weight)
    print('Episode {}: {}'.format(episode + 1, total_reward))
    total_rewards_eval.append(total_reward)
print('Average total reward over {} episode: {}'.format(n_episode_eval, sum(total_rewards_eval) / n_episode_eval))
#Average total reward over 100 episode: 198.66
