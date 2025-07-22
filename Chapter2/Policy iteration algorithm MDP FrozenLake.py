import torch
import gym

'''
Initializes a random policy. 
Computes the values of the policy with the policy evaluation algorithm. 
Obtains an improved policy based on the policy values. 
If the new policy is different from the old one, it updates the policy and runs another iteration. 
Otherwise, it terminates the iteration process and returns the policy values and the policy.
'''

def policy_evaluation(env,policy,gamma,threshold):
    env.reset()
    n_action = env.action_space.n
    n_state = env.observation_space.n
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(n_state):
            action = policy[state].item()
            for prob,new_state,reward,_ in env.env.P[state][action]:
                V_temp[state] += prob * (reward + gamma * V[new_state])
        delta_max = torch.max(torch.abs(V-V_temp))
        V = V_temp.clone()
        if delta_max <= threshold:
            break
    return V

def policy_improvement(env,V,gamma):
    env.reset()
    n_action = env.action_space.n
    n_state = env.observation_space.n
    policy = torch.zeros(n_state)
    for state in range(n_state):
        v_action = torch.zeros(n_action)
        for action in range(n_action):
            for prob,new_state,reward,_ in env.env.P[state][action]:
                v_action[action] += prob * (reward + gamma * V[new_state])
        policy[state] = torch.argmax(v_action)
    return policy

def policy_iteration(env,gamma,threshold):
    n_action = env.action_space.n
    n_state = env.observation_space.n
    optimal_policy = torch.randint(high=n_action,size=(n_state,)).float()

    while True:
        V = policy_evaluation(env,optimal_policy,gamma,threshold)
        time_policy = policy_improvement(env,V,gamma)
        if torch.equal(optimal_policy,time_policy):
            return V,optimal_policy
        optimal_policy = time_policy.clone()

env = gym.make('FrozenLake-v1',render_mode='rgb_array')#FrozenLake8x8-v1
n_action = env.action_space.n
n_state = env.observation_space.n

gamma = 0.99
threshold = 1e-4

v_optimal,optimal_policy = policy_iteration(env,gamma,threshold)
print('Optimal values:\n{}'.format(v_optimal))
print('Optimal policy:\n{}'.format(optimal_policy))


