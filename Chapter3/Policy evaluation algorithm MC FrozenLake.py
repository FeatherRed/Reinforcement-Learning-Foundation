import torch
import gym

def run_episode(env,policy):
    state = env.reset()[0]
    rewards = []
    states = [state]
    is_done_1 = False
    is_done_2 = False
    while not is_done_1 and not is_done_2:
        action = policy[state].item()
        state,reward,is_done_1,is_done_2,_ = env.step(action)
        states.append(state)
        rewards.append(reward)
    states = torch.tensor(states)
    rewards = torch.tensor(rewards)
    return states,rewards

def mc_prediction_first_visit(env,policy,gamma,n_episode):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    N = torch.zeros(n_state)
    for episode in range(n_episode):
        states_t,rewards_t = run_episode(env,policy)
        return_t = 0
        first_visit = torch.zeros(n_state)
        G = torch.zeros(n_state)
        for state_t,reward_t in zip(reversed(states_t)[1:],reversed(rewards_t)):
            return_t = gamma * return_t + reward_t
            G[state_t] = return_t
            first_visit[state_t] = 1
        for state in range(n_state):
            if first_visit[state] > 0:
                V[state] += G[state]
                N[state] += 1
    for state in range(n_state):
        if(N[state]) > 0:
            V[state] = V[state] / N[state]
    return V

def mc_prediction_every_visit(env,policy,gamma,n_episode):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    N = torch.zeros(n_state)
    G = torch.zeros(n_state)
    for episode in range(n_episode):
        states_t,rewards_t = run_episode(env,policy)
        return_t = 0
        for state_t,reward_t in zip(reversed(states_t)[1:],reversed(rewards_t)):
            return_t = gamma * return_t + reward_t
            G[state_t] += return_t
            N[state_t] += 1
    for state in range(n_state):
        if N[state] > 0:
            V[state] = G[state] / N[state]
    return V


env = gym.make('FrozenLake-v1',render_mode='rgb_array')
gamma = 1
n_episode = 10000
optimal_policy = torch.tensor([0., 3., 3., 3., 0., 3., 2., 3., 3., 1., 0., 3., 3., 2., 1., 3.])
value = mc_prediction_first_visit(env,optimal_policy,gamma,n_episode)
Value = mc_prediction_every_visit(env,optimal_policy,gamma,n_episode)
print('The value function calculated by first-visit MC prediction:\n', value)
print('The value function calculated by every-visit MC prediction:\n', Value)
'''
The value function calculated by first-visit MC prediction:
 tensor([0.7391, 0.4912, 0.4868, 0.4439, 0.7391, 0.0000, 0.3824, 0.0000, 0.7391,
        0.7401, 0.6688, 0.0000, 0.0000, 0.8003, 0.8922, 0.0000])
The value function calculated by every-visit MC prediction:
 tensor([0.6291, 0.4281, 0.3966, 0.3477, 0.6275, 0.0000, 0.3586, 0.0000, 0.6462,
        0.6767, 0.6353, 0.0000, 0.0000, 0.7674, 0.8772, 0.0000])

'''
