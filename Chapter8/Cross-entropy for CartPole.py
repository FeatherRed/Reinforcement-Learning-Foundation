import gym
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
class Estimator():
    def __init__(self, n_state, lr = 0.001):
        self.model = nn.Sequential(
            nn.Linear(n_state, 1),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def predict(self, s):
        return self.model(torch.Tensor(s))

    def update(self, s, y):
        y_pred = self.predict(s)
        y_pred = y_pred.reshape(-1)
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def cross_entropy(env, estimator, n_episode, n_samples):
    experience = []
    for episode in range(n_episode):
        rewards = 0
        actions = []
        states = []
        state = env.reset()[0]
        is_done_1 = False
        is_done_2 = False
        while not is_done_1 and not is_done_2:
            action = env.action_space.sample()
            states.append(state)
            next_state, reward, is_done_1, is_done_2, info = env.step(action)
            actions.append(action)
            rewards += reward

            state = next_state
        for state, action in zip(states, actions):
            experience.append((rewards, state, action))
    experience = sorted(experience, key = lambda x: x[0], reverse = True)
    select_experience = experience[: n_samples]
    train_states = [exp[1] for exp in select_experience]
    train_actions = [exp[2] for exp in select_experience]
    for _ in range(100):
        estimator.update(train_states, train_actions)

env = gym.make('CartPole-v0', render_mode = 'rgb_array')
n_state = env.observation_space.shape[0]
lr = 0.03

estimator = Estimator(n_state, lr)
n_episode = 5000
n_samples = 10000
cross_entropy(env, estimator, n_episode, n_samples)

n_episode = 100
total_reward_episode = [0] * n_episode
for episode in range(n_episode):
    state = env.reset()[0]
    is_done_1 = False
    is_done_2 = False
    while not is_done_1 and not is_done_2:
        action = 1 if estimator.predict(state).item() >= 0.5 else 0
        next_state, reward, is_done_1, is_done_2, info = env.step(action)
        total_reward_episode[episode] += reward
        state = next_state

plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
