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


bandit_payout_machines = [
    [0.01, 0.015, 0.03],
    [0.025, 0.01, 0.015]
]
bandit_reward_machines = [
    [1, 1, 1],
    [1, 1, 1]
]

n_machine = len(bandit_payout_machines)
env = [BanditEnv(bandit_payout, bandit_reward)
       for bandit_payout, bandit_reward in
       zip(bandit_payout_machines, bandit_reward_machines)]

n_episode = 100000
n_action = len(bandit_payout_machines[0])
action_count = torch.zeros(n_machine, n_action)
action_total_reward = torch.zeros(n_machine, n_action)
action_avg_reward = [[[] for action in range(n_action)] for _ in range(n_machine)]
Q = torch.zeros(n_machine, n_action)
ucb_policy = gen_ucb_policy(n_action)

for episode in range(n_episode):
    state = torch.randint(0, n_machine, (1,)).item()
    action = ucb_policy(Q[state], episode, action_count[state])
    reward = env[state].step(action)
    action_count[state][action] += 1
    action_total_reward[state][action] += reward
    Q[state][action] = action_total_reward[state][action] / action_count[state][action]
    for a in range(n_action):
        if action_count[state][a]:
            action_avg_reward[state][a].append(
                action_total_reward[state][a] / action_count[state][a]
            )
        else:
            action_avg_reward[state][a].append(0)

for state in range(n_machine):
    for action in range(n_action):
        plt.plot(action_avg_reward[state][action])
    plt.legend(['Arm {}'.format(action) for action in range(n_action)])
    plt.title('Average reward over time for state {}'.format(state))
    plt.xscale('log')
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.show()

