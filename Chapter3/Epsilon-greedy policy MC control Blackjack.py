import torch
import gym
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def run_episode(env,Q,epsilon,n_action):
    state = env.reset()[0]
    states = []
    actions = []
    rewards = []

    is_done_1 = False
    is_done_2 = False
    while not is_done_1 and not is_done_2:
        prob = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        prob[best_action] += 1.0 - epsilon
        action = torch.multinomial(prob,1).item()
        '''
        torch.multinomial(input,num_samples,replacement=False)
        输入一个权重张量input,每一行做一次n_samples取值，取出来的是对应下标
        replacement为是否放回，默认不返回
        '''
        states.append(state)
        actions.append(action)
        state,reward,is_done_1,is_done_2,_ = env.step(action)
        rewards.append(reward)
    return states,actions,rewards

def mc_control_epsilon_greedy(env,gamma,n_episode,epsilon):
    n_action = env.action_space.n
    V = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.empty(n_action))
    #Q state为字典 其key为torch.empty(n_action)
    policy = {}
    for episode in range(n_episode):
        states_t,actions_t,rewards_t = run_episode(env,Q,epsilon,n_action)
        return_t = 0
        G = {}
        for state_t,action_t,reward_t in zip(states_t[::-1],actions_t[::-1],rewards_t[::-1]):
            '''
            从后往前做回报
            '''
            return_t = return_t * gamma + reward_t
            G[(state_t,action_t)] = return_t
        for state_action,reward in G.items():
            state,action = state_action
            if state[0] <= 21:
                V[state_action] += reward
                N[state_action] += 1
                Q[state][action] = V[state_action] / N[state_action]
    for state,actions in Q.items():
    #state为一个字典，actions为torch tensor
        policy[state] = torch.argmax(actions).item()
    return Q,policy

def plot_surface(X,Y,Z,title):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111,projection='3d')
    surf = ax.plot_surface(X,Y,Z,rstride=1,cstride=1,
                           cmap=matplotlib.cm.coolwarm,vmin=-1.0,vmax=1.0)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    fig.colorbar(surf)
    plt.show()

def plot_blackjack_value(V):
    player_sum_range = range(12,22)
    dealer_show_range = range(1,11)
    X,Y = torch.meshgrid([torch.tensor(player_sum_range),torch.tensor(dealer_show_range)])
    values_to_plot = torch.zeros((len(player_sum_range),len(dealer_show_range),2))
    for i,player in enumerate(player_sum_range):
        for j,dealer in enumerate(dealer_show_range):
            for k,ace in enumerate([False,True]):
                values_to_plot[i,j,k] = V[(player,dealer,ace)]  #V是字典，key查询
    plot_surface(X,Y,values_to_plot[:,:,0].numpy(),'Blackjack Value Function Without Usable Ace')
    plot_surface(X, Y, values_to_plot[:, :, 1].numpy(), 'Blackjack Value Function With Usable Ace')

def simulate_episode(env,policy):
    state = env.reset()[0]
    rewards = 0
    is_done_1 = False
    is_done_2 = False
    while not is_done_1 and not is_done_2:
        action = policy[state]
        state,reward,is_done_1,is_done_2,_ = env.step(action)
        rewards += reward
    return rewards


env = gym.make('Blackjack-v1',render_mode='rgb_array')
gamma = 1
n_episode = 500000
epsilon = 0.1
optimal_Q, optimal_policy = mc_control_epsilon_greedy(env,gamma,n_episode,epsilon)
optimal_value = defaultdict(float)
for state,action_values in optimal_Q.items():
    optimal_value[state] = torch.max(action_values).item()
#plot_blackjack_value(optimal_value)

n_episode = 100000
n_win_optimal = 0
n_lose_optimal = 0
for episode in range(n_episode):
    reward = simulate_episode(env,optimal_policy)
    if reward == 1:
        n_win_optimal += 1
    elif reward == -1:
        n_lose_optimal += 1
print('Winning probability under the optimal policy: {}'.format(n_win_optimal/n_episode))
print('Losing probability under the optimal policy: {}'.format(n_lose_optimal/n_episode))
'''
Winning probability under the optimal policy: 0.43275
Losing probability under the optimal policy: 0.47742
'''
