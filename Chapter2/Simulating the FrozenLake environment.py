import torch
import gym

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

env = gym.make('FrozenLake-v1',render_mode='rgb_array')
n_action = env.action_space.n
n_state = env.observation_space.n

n_episode = 1000
total_rewards = []
for episode in range(n_episode):
    random_policy = torch.randint(high=n_action,size=(n_state,))
    #随机产生整数
    '''
        low 表示最低 默认 0
        high 表示最高 必须填值 
        size为规模
        范围是low~high-1
        tensor([3, 0, 3, 0, 0, 0, 2, 1, 3, 0, 1, 1, 3, 1, 3, 1])
    '''
    total_reward = run_episode(env,random_policy)
    print('Episode {}: {}'.format(episode + 1, total_reward))
    total_rewards.append(total_reward)

print('Average total reward over {} episode: {}'.format(n_episode, sum(total_rewards) / n_episode))
