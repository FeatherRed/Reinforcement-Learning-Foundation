import gym
import torch
import random
from collections import deque
import copy
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from CNN import DQN
image_size = 84
transform = T.Compose(
    [T.ToPILImage(),
     T.Resize((image_size, image_size),
              interpolation = Image.CUBIC),
     T.ToTensor()
     ]
)
def get_state(obs):
    state = obs.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    state = transform(state).unsqueeze(0)
    return state

def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function

def q_learning(env, estimator, n_episode, replay_size, target_update = 10, gamma = 1.0, epsilon = 0.1, epsilon_decay = 0.99):
    for episode in range(n_episode):
        if episode % target_update == 0:
            estimator.copy_target()
        obs = env.reset()[0]
        state = get_state(obs)
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        is_done_1 = False
        is_done_2 = False
        while not is_done_1 and not is_done_2:
            action = policy(state)
            next_obs, reward, is_done_1, is_done_2, info = env.step(ACTIONS[action])
            total_reward_episode[episode] += reward
            next_state = get_state(next_obs)

            memory.append((state, action, next_state, reward, is_done_1, is_done_2))
            if is_done_1 or is_done_2:
                break
            estimator.replay(memory, replay_size, gamma)
            state = next_state
        print('Episode: {}, total reward: {}, epsilon:{}'.format(episode, total_reward_episode[episode], epsilon))
        epsilon = max(epsilon * epsilon_decay, 0.01)


env = gym.envs.make("PongDeterministic-v4", render_mode = 'rgb_array')
ACTIONS = [0, 2, 3]
n_action = 3
state_shape = env.observation_space.shape
n_state = image_size * image_size
n_hidden = [200, 50]
n_episode = 1000
lr = 0.00025
replay_size = 32
target_update = 10
memory = deque(maxlen = 100000)
total_reward_episode = [0] * n_episode
dqn = DQN(3, n_action, lr)

q_learning(env, dqn, n_episode, replay_size, target_update)