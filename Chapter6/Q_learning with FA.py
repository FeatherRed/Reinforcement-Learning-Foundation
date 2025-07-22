import gym
import torch
from Linear_estimator import Estimator
import matplotlib.pyplot as plt

def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        probs = torch.ones(n_action) * epsilon / n_action
        q_values = estimator.predict(state)
        best_action = torch.argmax(q_values).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function

def q_learning(env, estimator, n_episode, gamma, epsilon, epsilon_decay):
    n_action = env.action_space.n
    for episode in range(n_episode):
        epsilon_greedy_policy = gen_epsilon_greedy_policy(estimator, epsilon * epsilon_decay ** epsilon, n_action)
        state = env.reset()[0]
        is_done_1 = False
        is_done_2 = False
        while not is_done_1 and not is_done_2:
            action = epsilon_greedy_policy(state)
            next_state, reward, is_done_1, is_done_2, info = env.step(action)
            q_value_next = estimator.predict(next_state)
            td_target = reward + gamma * torch.max(q_value_next)
            estimator.update(state, action, td_target)
            total_reward_episode[episode] += reward

            state = next_state

env = gym.make('MountainCar-v0',render_mode = 'rgb_array')
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_feature = 200
lr = 0.03
estimator = Estimator(n_feature, n_state, n_action, lr)
n_episode = 300
total_reward_episode = [0] * n_episode
gamma = 1.0
epsilon = 0.1
epsilon_decay = 0.99
q_learning(env, estimator, n_episode, gamma, epsilon, epsilon_decay)

plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
