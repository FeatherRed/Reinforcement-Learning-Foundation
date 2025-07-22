import gym
import torch
from collections import deque
import random
from torch.autograd import Variable
from Double_DQN import DQN
import matplotlib.pyplot as plt
def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function

def q_learning(env, estimator, n_episode, replay_size, target_update ,gamma, epsilon, epsilon_decay):
    n_action = env.action_space.n
    n_state = env.observation_space.shape[0]
    for episode in range(n_episode):
        if episode % target_update == 0:
            estimator.copy_target()
        state = env.reset()[0]
        is_done_1 = False
        is_done_2 = False
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        while not is_done_1 and not is_done_2:
            action = policy(state)
            next_state, reward, is_done_1, is_done_2, info = env.step(action)
            total_reward_episode[episode] += reward
            modified_reward = next_state[0] + 0.5
            if next_state[0] >= 0.5:
                modified_reward += 100
            elif next_state[0] >= 0.25:
                modified_reward += 20
            elif next_state[0] >= 0.1:
                modified_reward += 10
            elif next_state[0] >= 0:
                modified_reward += 5

            memory.append((state, action, next_state, modified_reward, is_done_1, is_done_2))

            if is_done_1 or is_done_2:
                break
            estimator.replay(memory, replay_size, gamma)
            state = next_state
        print('Episode: {}, total reward: {}, epsilon:{}'.format(episode, total_reward_episode[episode], epsilon))
        epsilon = max(epsilon * epsilon_decay, 0.01)

env = gym.make("MountainCar-v0", render_mode = "rgb_array")
n_action = env.action_space.n
n_state = env.observation_space.shape[0]
n_episode = 1000
replay_size = 20
target_update = 10
total_reward_episode = [0] * n_episode
memory = deque(maxlen = 10000)
dqn = DQN(n_state, n_action, lr = 0.01)
q_learning(env, dqn, n_episode, replay_size, target_update, gamma = 0.9, epsilon = 1, epsilon_decay = 0.99)

plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()