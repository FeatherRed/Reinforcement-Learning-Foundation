import gym
import torch
from nn_estimator import Estimator
from collections import deque
import random
import matplotlib.pyplot as plt
def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        probs = torch.ones(n_action) * epsilon / n_action
        q_value = estimator.predict(state)
        best_action = torch.argmax(q_value).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function

def q_learning(env, estimator, n_episode, replay_size, gamma, epsilon, epsilon_decay):
    n_action = env.action_space.n
    for episode in range(n_episode):
        state = env.reset()[0]
        epsilon_greedy_policy = gen_epsilon_greedy_policy(estimator, epsilon * epsilon_decay ** epsilon, n_action)
        is_done_1 = False
        is_done_2 = False
        while not is_done_1 and not is_done_2:
            action = epsilon_greedy_policy(state)
            next_state, reward, is_done_1, is_done_2, info = env.step(action)
            total_reward_episode[episode] += reward
            if is_done_1 or is_done_2:
                break

            q_value_next = estimator.predict(next_state)
            td_target = reward + gamma * torch.max(q_value_next)
            memory.append((state, action, td_target))
            state = next_state
        replay_data = random.sample(memory, min(replay_size, len(memory)))

        for state, action, td_target in replay_data:
            estimator.update(state, action, td_target)


env = gym.make('MountainCar-v0', render_mode = 'rgb_array')
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_feature = 200
n_hidden = 50
lr = 0.001
estimator = Estimator(n_feature, n_state, n_action, n_hidden, lr)
memory = deque(maxlen = 300)
n_episode = 1000
total_reward_episode = [0] * n_episode
gamma = 1.0
epsilon = 0.1
epsilon_decay = 0.99
replay_size = 200

q_learning(env, estimator, n_episode, replay_size, gamma, epsilon, epsilon_decay)

plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()