import gym
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

def gen_epsilon_greedy_policy(n_action, epsilon):
    def policy_function(state, Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function

def sarsa(env, gamma, n_episode, alpha):
    n_action = env.action_space.n
    n_state = env.observation_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in range(n_episode):
        state = env.reset()[0]
        is_done_1 = False
        is_done_2 = False
        action = epsilon_greedy_policy(state, Q)
        while not is_done_1 and not is_done_2:
            next_state, reward, is_done_1, is_done_2, info = env.step(action)
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            next_action = epsilon_greedy_policy(next_state, Q)
            td_delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            Q[state][action] += alpha * td_delta

            state = next_state
            action = next_action
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy



env = gym.make("Taxi-v3",render_mode="rgb_array")
n_state = env.observation_space.n
n_action = env.action_space.n
n_episode = 1000
length_episode = [0] * n_episode
total_reward_episode = [0] * n_episode
gamma = 1
alpha = 0.4
epsilon = 0.01
epsilon_greedy_policy = gen_epsilon_greedy_policy(n_action, epsilon)

optimal_Q, optimal_policy = sarsa(env, gamma, n_episode, alpha)
print('The optimal policy:\n', optimal_policy)

plt.plot(length_episode)
plt.title('Episode length over time')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.show()
plt.plot(total_reward_episode)
plt.title('Episode length over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()

'''Env = gym.make("Taxi-v3",render_mode="human")
state = Env.reset()[0]
is_done_1 = False
is_done_2 = False
while not is_done_1 and not is_done_2:
    action = optimal_policy[state]
    state, reward, is_done_1, is_done_2, info = Env.step(action)'''