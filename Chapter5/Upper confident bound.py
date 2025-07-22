import torch
from Multi_armed_bandit import BanditEnv
import matplotlib.pyplot as plt

def gen_ucb_policy(n_action):
    def policy_function(Q, episode, count):
        ucb = Q.clone()
        ucb += torch.sqrt(2 * torch.log(torch.ones(n_action) * episode) / count)
        action = torch.argmax(ucb).item()
        return action
    return policy_function


bandit_payout = [0.1, 0.15, 0.3]
bandit_reward = [4, 3, 1]
env = BanditEnv(bandit_payout,bandit_reward)
n_episode = 100000
n_action = len(bandit_payout)
action_count = [0 for _ in range(n_action)]
action_total_reward = [0 for _ in range(n_action)]
action_avg_reward = [[] for action in range(n_action)]
Q = torch.zeros(n_action)
ucb_policy = gen_ucb_policy(n_action)

for episode in range(n_episode):
    action = ucb_policy(Q, episode, torch.IntTensor(action_count))
    reward = env.step(action)

    action_count[action] += 1
    action_total_reward[action] += reward
    Q[action] = action_total_reward[action] / action_count[action]
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