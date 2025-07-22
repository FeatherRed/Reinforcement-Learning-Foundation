import gym
import torch
from PolicyNetwork import PolicyNetwork, ValueNetwork
import matplotlib.pyplot as plt
def reinforce(env, estimator_policy, estimator_value, n_episode, gamma = 1.0):
    for episode in range(n_episode):
        log_probs = []
        rewards = []
        states = []
        state = env.reset()[0]
        is_done_1 = False
        is_done_2 = False
        while not is_done_1 and not is_done_2:
            states.append(state)
            action, log_prob = estimator_policy.get_action(state)
            next_state, reward, is_done_1, is_done_2, info = env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        Gt = 0
        pw = 0
        returns = []
        for t in range(len(states)-1, -1, -1):
            Gt += gamma ** pw * rewards[t]
            pw += 1
            returns.append(Gt)
        returns = returns[::-1]
        returns = torch.tensor(returns)
        baseline_value = estimator_value.predict(states)
        advantages = returns - baseline_value
        estimator_value.update(states, returns)
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        estimator_policy.update(advantages, log_probs)
        print('Episode: {}, total reward: {}'.format(episode, total_reward_episode[episode]))

env = gym.make('CartPole-v0', render_mode = 'rgb_array')
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_hidden_p = 64
n_hidden_v = 64
lr_p = 0.003
lr_v = 0.003
gamma = 0.9
n_episode = 2000
policy_net = PolicyNetwork(n_state, n_action, n_hidden_p, lr_p)
value_net = ValueNetwork(n_state, n_hidden_v, lr_v)
total_reward_episode = [0] * n_episode
reinforce(env, policy_net, value_net, n_episode, gamma)

plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()

'''Env = gym.make('CartPole-v0', render_mode = 'human')
for episode in range(n_episode):
    state = Env.reset()[0]
    is_done_1 = False
    is_done_2 = False
    while not is_done_1 and not is_done_2:
        action, _ = policy_net.get_action(state)
        next_state, reward, is_done_1, is_done_2, info = Env.step(action)
        state = next_state'''

