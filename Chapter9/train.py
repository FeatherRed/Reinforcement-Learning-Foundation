import random
import torch
from collections import deque
import flappybird as bird
from DQN_for_FlappyBird import DQN
import cv2
import numpy as np
import time
from tqdm import tqdm

def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function

def pre_processing(image, width, height):
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image[None, :, :].astype(np.float32)


if __name__ == "__main__":
    image_size = 84
    batch_size = 32
    lr = 1e-6
    gamma = 0.99
    init_epsilon = 0.1
    final_epsilon = 1e-4
    n_iter = 2000000
    memory_size = 50000
    n_action = 2
    save_path = 'trained_models'
    torch.manual_seed(123)
    estimator = DQN(n_action)
    memory = deque(maxlen = memory_size)
    env = bird.FlappyBird()
    image, reward, is_done = env.step(0)
    image = pre_processing(image[:bird.screen_width, :int(env.bird_y)], image_size, image_size)
    image = torch.from_numpy(image)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    with tqdm(range(n_iter)) as pbar:
        for iter in range(n_iter):
            epsilon = final_epsilon + (n_iter - iter) * (init_epsilon - final_epsilon) / n_iter
            policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
            action = policy(state)
            next_image, reward, is_done = env.step(action)
            next_image = pre_processing(next_image[:bird.screen_width, :int(env.base_y)], image_size, image_size)
            next_image = torch.from_numpy(next_image)
            next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
            memory.append([state, action, next_state, reward, is_done])
            loss = estimator.replay(memory, batch_size, gamma)
            state = next_state
            pbar.set_postfix(Action = action, Loss = loss, Epsilon = epsilon, Reward = reward)
            pbar.update(1)
            # print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}".format(
            #     iter + 1, n_iter, action, loss, epsilon, reward))
            if (iter + 1) % 10000 == 0:
                torch.save(estimator.model, "{}/{}".format((save_path, iter + 1)))
