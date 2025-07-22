'''
    该文件是求随机搜索策略达到200时的轮次
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
n_action = env.action_space.n
n_state = env.observation_space.shape[0]

n_episode = 1000
best_total_reward = 0
best_weight = None
total_rewards = []

for episode in range(n_episode):
    weight = torch.rand(n_state,n_action)
    total_reward = run_episode(env,weight)
    print('Episode {}: {}'.format(episode + 1, total_reward))
    if best_total_reward < total_reward:
        best_total_reward = total_reward
        best_weight = weight
    total_rewards.append(total_reward)
    if best_total_reward == 200:
        break

n_training = 1000
n_episode_training = []
for _ in range(n_training):
    for episode in range(n_episode):
        weight = torch.rand(n_state, n_action)
        total_reward = run_episode(env, weight)
        if total_reward == 200:
            n_episode_training.append(episode+1)
            break
print('Expectation of training episodes needed: ', sum(n_episode_training) / n_training)