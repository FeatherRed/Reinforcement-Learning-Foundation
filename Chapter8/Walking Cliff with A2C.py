import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
class ActorCriticModel(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(ActorCriticModel, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden[0])
        self.fc2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.action = nn.Linear(n_hidden[1], n_output)
        self.value = nn.Linear(n_hidden[1], 1)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action(x), dim = -1)
        state_values = self.value(x)
        return action_probs, state_values

class PolicyNetwork():
    def __init__(self, n_state, n_action, n_hidden, lr = 0.001):
        self.model = ActorCriticModel(n_state, n_action, n_hidden)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 10, gamma = 0.9)

    def predict(self, s):
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
        action_probs, state_value = self.predict(s)
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[action])
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
            action, log_prob, state_value = estimator.get_action(state)
            next_state, reward, is_done_1, is_done_2, info = env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            rewards.append(reward)
            state_values.append(state_value)

            state = next_state
        Gt = 0
        returns = []
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
        if total_reward_episode[episode] >= 195:
            estimator.scheduler.step()

def actor_critic(env, estimator, n_episode, gamma = 1.0):
    for episode in range(n_episode):
        log_probs = []
        rewards = []
        state_values = []
        state = env.reset()[0]
        is_done_1 = False
        is_done_2 = False
        while not is_done_1 and not is_done_2:
            one_hot_state = [0] * 48
            one_hot_state[state] = 1
            action, log_prob, state_value = estimator.get_action(one_hot_state)
            next_state ,reward, is_done_1, is_done_2, info = env.step(action)
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
        if total_reward_episode[episode] >= -14:
            estimator.scheduler.step()


env = gym.make('CliffWalking-v0', render_mode = 'rgb_array')
n_action = env.action_space.n
n_state = 48
n_hidden = [128, 32]
n_episode = 1000
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

