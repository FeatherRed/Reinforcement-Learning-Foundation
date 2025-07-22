import torch
import gym
import random
import torchvision.transforms as T
from PIL import Image
from Double_DQN_Atari import DQN
from collections import deque
import matplotlib.pyplot as plt
image_size = 84
transform = T.Compose(
    [T.ToPILImage(),
     T.Grayscale(num_output_channels = 1),
     T.Resize((image_size,image_size),
              interpolation = Image.CUBIC),
     T.ToTensor(),
     ]
)

def get_state(obs):
    state = obs.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    state = transform(state)
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
    n_action = env.action_space.n
    for episode in range(n_episode):
        if episode % target_update == 0:
            estimator.copy_target()
        obs = env.reset()[0]
        state = get_state(obs).view(image_size * image_size)
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        is_done_1 = False
        is_done_2 = False
        while not is_done_1 and not is_done_2:
            action = policy(state)
            next_obs, reward, is_done_1, is_done_2, info = env.step(action)
            total_reward_episode[episode] += reward
            next_state = get_state(next_obs).view(image_size * image_size)

            memory.append((state, action, next_state, reward, is_done_1, is_done_2))
            if is_done_1 or is_done_2:
                break
            estimator.replay(memory, replay_size, gamma)
            state = next_state
        print('Episode: {}, total reward: {}, epsilon:{}'.format(episode, total_reward_episode[episode], epsilon))
        epsilon = max(epsilon * epsilon_decay, 0.01)


env = gym.make("PongDeterministic-v4", render_mode = "rgb_array")
state_shape = env.observation_space.shape
n_action = env.action_space.n
n_state = image_size * image_size
n_hidden = [200, 50]
# obs = env.reset()[0]
# state = get_state(obs)
# print(state.shape)

n_episode = 1000
lr = 0.003
replay_size = 32
target_update = 10
memory = deque(maxlen = 10000)
total_reward_episode = [0] * n_episode
dqn = DQN(n_state, n_action, n_hidden, lr)
q_learning(env, dqn, n_episode, replay_size, target_update, gamma = .9, epsilon = 1)

plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()