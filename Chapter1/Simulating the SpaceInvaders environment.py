import torch
import gym

env = gym.make('SpaceInvaders-v0',render_mode='human')
env.reset()
is_done_1 = False
is_done_2 = False

while not is_done_1 and not is_done_2:
    action = env.action_space.sample()
    new_state,reward,is_done_1,is_done_2,info = env.step(action)
    print(info)
    env.render()
