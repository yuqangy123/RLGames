import gym.envs
import torch
import gym
from ddpg import ReplayBuffer, DDPG

_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -------------------------------------- #
# 环境加载
# -------------------------------------- #
env_name = "MountainCarContinuous-v0"  # 连续型动作
env = gym.make(env_name)
env.seed(0)
n_states = env.observation_space.shape[0]   # 状态数
n_actions = env.action_space.shape[0]       # 动作数
action_bound = env.action_space.high[1.0]   # 动作值边界

# -------------------------------------- #
# 模型构建， 回放经验池
# -------------------------------------- #
replay_buffer = ReplayBuffer(capacity=100000)
agent = DDPG(n_states, n_actions, hiddens=256, action_bound=action_bound, 
            sigma=0.05, gamma=0.99, tau=0.001, lr_a=0.001, lr_c=0.001, device=_device)


# -------------------------------------- #
# 模型训练
# -------------------------------------- #
return_list = []        #每个回合的return
mean_return_list = []   #每个回合的mean_return均值

for i_episode in range(100):
    episode_return = 0  #累计每条链上的reward
    state = env.reset() #初始化状态
    env.render()        #gui显示游戏
    done = False        #游戏是否结束

    while not done:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)

        #加入经验回放池
        replay_buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_return += reward

        if replay_buffer.size() > 3:  #当经验池中有3条经验时，开始训练
            #随机采样2条数据
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(2)
            #构建数据集

            agent.update(b_s, b_a, b_r, b_ns, b_d)



