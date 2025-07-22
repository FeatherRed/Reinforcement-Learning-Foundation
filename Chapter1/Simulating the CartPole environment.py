import gym
from gym.wrappers import RecordVideo

# 创建环境并指定视频保存目录
env = gym.make('CartPole-v0',render_mode = 'rgb_array')
video_dir = './cartpole_video/'

# 在环境上包装视频记录
env = RecordVideo(env,video_dir)

# 重置环境
env.reset()

is_done_1 = False
is_done_2 = False

while not is_done_1 and not is_done_2:
    action = env.action_space.sample()
    new_state, reward, is_done_1, is_done_2, info = env.step(action)
    # 显示环境渲染结果
env.close()
