import torch
from Multi_armed_bandit import BanditEnv
import matplotlib.pyplot as plt

def gen_thompson_policy():
    def policy_function(alpha, beta):
        prior_values = torch.distributions.beta.Beta(alpha, beta).sample()
        return torch.argmax(prior_values).item()
    return policy_function


bandit_payout = [0.1, 0.15, 0.3]
bandit_reward = [1, 1, 1]
env = BanditEnv(bandit_payout,bandit_reward)
n_episode = 100000
n_action = len(bandit_payout)
action_count = [0 for _ in range(n_action)]
action_total_reward = [0 for _ in range(n_action)]
action_avg_reward = [[] for action in range(n_action)]
thompson_policy = gen_thompson_policy()
alpha = torch.ones(n_action)
beta = torch.ones(n_action)


for episode in range(n_episode):
    action = thompson_policy(alpha, beta)
    reward = env.step(action)
    action_count[action] += 1
    action_total_reward[action] += reward
    if reward:
        alpha[action] += 1
    else:
        beta[action] += 1
    for a in range(n_action):
        if action_count[a]:
            action_avg_reward[a].append(
                action_total_reward[a] / action_count[a]
            )
        else:
            action_avg_reward[a].append(0)

for action in range(n_action):
    plt.plot(action_avg_reward[action])
plt.legend(['Arm {}'.format(action) for action in range(n_action)])
plt.title('Average reward over time')
plt.xscale('log')
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.show()
