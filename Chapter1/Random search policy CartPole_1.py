import torch
import gym
import matplotlib.pyplot as plt


env = gym.make('CartPole-v0',render_mode='rgb_array')
n_state = env.observation_space.shape[0]
n_action = env.action_space.n

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

#main
n_episode = 1000
best_reward = 0
best_weight = []
total_reward = []

for episode in range(n_episode):
    env.reset()
    weight = torch.rand(n_state,n_action)
    reward = run_episode(env,weight)
    #print('Episode {}: {}'.format(episode + 1, reward))
    if reward > best_reward:
        best_reward = reward
        best_weight = weight
    total_reward.append(reward)

#print('Average total reward over {} episode: {}'.format(n_episode, sum(total_reward) / n_episode))

##用随机得到的weight运行

n_episode_test = 100
total_reward_test = []
for episode in range(n_episode_test):
    env.reset()
    reward = run_episode(env,best_weight)
    print('Episode {}: {}'.format(episode + 1, reward))
    total_reward_test.append(reward)

print('Average total reward over {} episode: {}'.format(n_episode_test, sum(total_reward_test) / n_episode_test))


##plot
plt.plot(total_reward)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
