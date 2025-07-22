import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing
import numpy as np
import matplotlib.pyplot as plt
class ActorCriticModel(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(ActorCriticModel, self).__init__()
        self.fc = nn.Linear(n_input, n_hidden)
        self.mu = nn.Linear(n_hidden, n_output)
        self.sigma = nn.Linear(n_hidden, n_output)
        self.value = nn.Linear(n_hidden, 1)
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2 * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 1e-5
        dist = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        value = self.value(x)
        return dist, value

class PolicyNetwork():
    def __init__(self, n_state, n_action, n_hidden = 50, lr = 0.001):
        self.model = ActorCriticModel(n_state, n_action, n_hidden)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def predict(self, s):
        self.model.training = False
        return self.model(torch.Tensor(s))

    def update(self, returns, log_probs, state_values):
        loss = 0
        for log_prob, value, Gt in zip(log_probs, state_values, returns):
            advantage = Gt - value.item()
            policy_loss = -log_prob * advantage
            value_loss = F.smooth_l1_loss(value, Gt)
            loss += policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, s):
        dist, state_value = self.predict(s)
        action = dist.sample()
        log_prob = dist.log_prob(action[0])
        return action, log_prob, state_value

def actor_critic(env, estimator, n_episode, gamma = 1.0):
    for episode in range(n_episode):
        log_probs = []
        rewards = []
        state_values = []
        state = env.reset()[0]
        is_done_1 = False
        is_done_2 = False
        while not is_done_1 and not is_done_2:
            state = scale_state(state)
            action, log_prob, state_value = estimator.get_action(state)
            action = action.clip(env.action_space.low[0], env.action_space.high[0])
            next_state, reward, is_done_1, is_done_2, info = env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            rewards.append(reward)
            state_values.append(state_value)
            state = next_state
        returns = []
        Gt = 0
        pw = 0
        for reward in rewards[::-1]:
            Gt += gamma ** pw * reward
            pw += 1
            returns.append(Gt)
        returns = returns[::-1]
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9) # 变正态分布
        estimator.update(returns, log_probs, state_values)
        print('Episode: {}, total reward: {}'.format(episode, total_reward_episode[episode]))

def scale_state(state):
    scaled = scaler.transform([state])
    return scaled[0]



env = gym.make('MountainCarContinuous-v0', render_mode = 'rgb_array')
state_space_samples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(state_space_samples)

n_state = env.observation_space.shape[0]
n_action = 1
n_hidden = 128
n_episode = 200
lr = 0.003
policy_net = PolicyNetwork(n_state, n_action, n_hidden, lr)
total_reward_episode = [0] * n_episode
gamma = 0.9
actor_critic(env, policy_net, n_episode, gamma)

plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()

