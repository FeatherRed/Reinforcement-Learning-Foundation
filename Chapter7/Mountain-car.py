import gym
import random
import torch
from DQN import DQN
import matplotlib.pyplot as plt
def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function

def q_learning(env, estimator, n_episode, gamma, epsilon, epsilon_decay):
    n_action = env.action_space.n
    n_state = env.observation_space.shape[0]
    for episode in range(n_episode):
        state = env.reset()[0]
        is_done_1 = False
        is_done_2 = False
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        while not is_done_1 and not is_done_2:
            action = policy(state)
            next_state, reward, is_done_1, is_done_2, info = env.step(action)
            total_reward_episode[episode] += reward
            modified_reward = next_state[0] + 0.5
            if next_state[0] >= 0.5:
                modified_reward += 100
            elif next_state[0] >= 0.25:
                modified_reward += 20
            elif next_state[0] >= 0.1:
                modified_reward += 10
            elif next_state[0] >= 0:
                modified_reward += 5

            q_values = estimator.predict(state).tolist()
            if is_done_1 or is_done_2:
                q_values[action] = modified_reward
                estimator.update(state, q_values)
                break
            q_values_next = estimator.predict(next_state)
            q_values[action] = modified_reward + gamma * torch.max(q_values_next).item()
            estimator.update(state, q_values)
            state = next_state
        print('Episode: {}, total reward: {}, epsilon:{}'.format(episode, total_reward_episode[episode], epsilon))
        epsilon = max(epsilon * epsilon_decay, 0.01)


env = gym.make("MountainCar-v0", render_mode = "rgb_array")
n_action = env.action_space.n
n_state = env.observation_space.shape[0]
n_episode = 1000
total_reward_episode = [0] * n_episode
dqn = DQN(n_state,n_action, lr = 0.001)
q_learning(env, dqn, n_episode, gamma = 0.99, epsilon = 0.3, epsilon_decay = 0.99)

plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
