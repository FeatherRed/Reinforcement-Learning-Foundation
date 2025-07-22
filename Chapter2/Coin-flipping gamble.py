import torch

def policy_evaluation(env,policy,gamma,threshold):
    capital = env['capital']
    n_state = env['n_state']
    rewards = env['rewards']
    prob = env['prob']
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(1,capital):
            action = policy[state].item()
            V_temp[state] += prob * (rewards[state + action] + gamma * V[state + action])
            V_temp[state] += (1 - prob) * (rewards[state - action] + gamma * V[state - action])
        delta_max = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if delta_max <= threshold:
            break
    return V

def policy_improvement(env,V,gamma):
    capital = env['capital']
    n_state = env['n_state']
    rewards = env['rewards']
    prob = env['prob']
    policy = torch.zeros(n_state).int()

    for state in range(1,capital):
        n_action = min(state,capital - state)
        v_action = torch.zeros(n_action + 1)
        for action in range(1,n_action + 1):
            v_action[action] += prob * (rewards[state + action] + gamma * V[state + action])
            v_action[action] += (1 - prob) * (rewards[state - action] + gamma * V[state - action])
        policy[state] = torch.argmax(v_action)
    return policy

def policy_iteration(env,gamma,threshold):
    capital = env['capital']
    n_state = env['n_state']
    rewards = env['rewards']
    prob = env['prob']
    optimal_policy = torch.zeros(n_state).int()

    while True:
        V = policy_evaluation(env,optimal_policy,gamma,threshold)
        temp_policy = policy_improvement(env,V,gamma)
        if torch.equal(optimal_policy,temp_policy):
            return V,optimal_policy
        optimal_policy = temp_policy.clone()


def optimal_strategy(capital):
    return optimal_policy[capital].item()
def conservative_strategy(capital):
    return 1
def random_strategy(capital):
    return torch.randint(1,capital + 1,(1,)).item()

def run_episode(head_prob,capital,policy):
    while capital > 0:
        bet = policy(capital)
        if torch.rand(1).item() < head_prob:
            capital += bet
            if capital >= 100:
                return 1
        else:
            capital -= bet
    return 0
'''
============================================================================================================
'''
gamma = 1
threshold = 1e-10
capital_max = 100
n_state = capital_max + 1
rewards = torch.zeros(n_state)
rewards[-1] = 1
head_prob = 0.4
env = {'capital':capital_max,
       'prob':head_prob,
       'rewards':rewards,
       'n_state':n_state}
V_optimal,optimal_policy = policy_iteration(env,gamma,threshold)
capital = 50
n_episode = 10000
n_win_random = 0
n_win_conservative = 0
n_win_optimal = 0
for episode in range(n_episode):
    n_win_optimal += run_episode(head_prob,capital,optimal_strategy)
    n_win_conservative += run_episode(head_prob,capital,conservative_strategy)
    n_win_random += run_episode(head_prob,capital,random_strategy)
print('Average winning probability under the optimal policy: {}'.format(n_win_optimal/n_episode))
print('Average winning probability under the conservative policy: {}'.format(n_win_conservative/n_episode))
print('Average winning probability under the random policy: {}'.format(n_win_random/n_episode))