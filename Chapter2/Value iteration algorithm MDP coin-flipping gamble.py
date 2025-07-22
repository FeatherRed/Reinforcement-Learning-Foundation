import torch
import time
import matplotlib.pyplot as plt

def value_iteration(env,gamma,threshold):
    n_state = env['n_state']
    capital = env['capital']
    rewards = env['rewards']
    prob = env['prob']
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(1,capital):
            n_action = min(state,capital - state)
            v_action = torch.zeros(n_action + 1)
            for action in range(1,n_action + 1):
                #两种情况 一个是s+a 一个是s-a
                v_action[action] += prob * (rewards[state + action] + gamma * V[state + action])
                v_action[action] += (1 - prob) * (rewards[state - action] + gamma * V[state - action])
            if v_action.numel():
                V_temp[state] = torch.max(v_action)
        delta_max = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if delta_max <= threshold:
            break
    return V

def extract_optimal_policy(env,V_optimal,gamma):
    n_state = env['n_state']
    capital = env['capital']
    rewards = env['rewards']
    prob = env['prob']
    policy = torch.zeros(capital).int()
    for state in range(1,capital):
        n_action = min(state,capital - state)
        v_action = torch.zeros(n_action + 1)
        for action in range(1,n_action + 1):
            v_action[action] += prob * (rewards[state + action] + gamma * V_optimal[state + action])
            v_action[action] += (1 - prob) * (rewards[state - action] + gamma * V_optimal[state - action])
        policy[state] = torch.argmax(v_action)
    return policy


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
start_time = time.time()
V_optimal = value_iteration(env,gamma,threshold)
optimal_policy = extract_optimal_policy(env,V_optimal,gamma)
print("It takes {:.3f}s to solve with value iteration".format(time.time() - start_time))
print('Optimal values:\n{}'.format(V_optimal))
print('Optimal policy:\n{}'.format(optimal_policy))

plt.plot(V_optimal[:100].numpy())
plt.title('Optimal policy values')
plt.xlabel('Capital')
plt.ylabel('Policy value')
plt.show()