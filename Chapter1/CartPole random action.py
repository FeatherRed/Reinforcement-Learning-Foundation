import gym


env = gym.make('CartPole-v0',render_mode='human')
n_episode = 10000
total_reward = []

for episode in range(n_episode):
    env.reset()
    is_done_1 = False
    is_done_2 = False
    rewards = 0
    while not is_done_1 and not is_done_2:
        action = env.action_space.sample()
        new_state,reward,is_done_1,is_done_2,info = env.step(action)
        rewards += reward
    total_reward.append(rewards)

print('Average total reward over {} episodes: {}'.format(n_episode, sum(total_reward) / n_episode))