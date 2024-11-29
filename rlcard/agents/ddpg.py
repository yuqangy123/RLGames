import collections
import random
import numpy as np
from torch import  nn
import torch
import torch.nn.functional as F


'''DDPG强化学习算法利用策略网络对当前状态做出动作值、
评价网络对给出的状态-动作对进行评价
并循环利用经验回放池对策略网络和价值网络进行更新、
软更新、参数更新
DDPG强化学习算法适合于连续的动作空间'''

#经验回放池
class ReplayBuffer():
    def __init__(self, capacity) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        return len(self.buffer)
    

#策略网络
class PolicyNet(nn.Module):
    def __init__(self, n_states, n_actions, hiddens, action_bound):
        super(PolicyNet, self).__init__()
        self.action_bound = action_bound        #连续型动作值的边界
        
        fc = [nn.Linear(n_states, hiddens),
              nn.ReLU(),
              nn.Linear(hiddens, n_actions),
              nn.Tanh()
              ]
        self.net = nn.Sequential(*fc)

    def forward(self, x):
        x = self.net(x)
        x *= self.action_bound
        return x
    

#价值网络
class ValueNet(nn.Module):
    def __init__(self, n_states, n_actions, hiddens):
        super(PolicyNet, self).__init__()
        fc = [nn.Linear(n_states+n_actions, hiddens),
              nn.ReLU(),
              nn.Linear(hiddens, hiddens),
              nn.ReLU(),
              nn.Linear(hiddens, 1),
              ]
        self.net = nn.Sequential(*fc)

    def forward(self, states, actions):
        x = self.net(torch.cat([states, actions], dim=1))
        return x

class DDPG():
    def __init__(self, n_states, n_actions, hiddens, action_bound, sigma, gamma, tau, lr_a, lr_c, device):
        self.n_states = n_states
        self.n_actions = n_actions
        self.hiddens = hiddens
        self.action_bound = action_bound
        self.gamma = gamma                  #折扣因子
        self.tau = tau                      #软更新调节系数
        self.sigma = sigma                  #动作噪声调节系数
        self.device = device

        #策略网络
        self.actor = PolicyNet(n_states, n_actions, hiddens, action_bound).to(device)
        self.actor_target = PolicyNet(n_states, n_actions, hiddens, action_bound).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a)

        #评价网络
        self.critic = ValueNet(n_states, n_actions, hiddens).to(device)
        self.critic_target = ValueNet(n_states, n_actions, hiddens).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        
    #动作值函数
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).view(-1, 1).to(self.device)
        action = self.actor(state)      #计算当前状态下的动作 [b, n_states] -> [b, action_bound]
        action = action.item() + self.sigma * np.random.randn(self.n_actions)   #加噪声，探索新的空间
        return action

    #软更新: 意思是每次更新的时候只更新部分参数
    def soft_update(self, net, target_net):
        #获取训练网络和目标网络的参数
        for param, param_target in zip(net.parameters(), target_net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(self.tau * param.data + (1 - self.tau) * param_target.data)

    #参数更新
    def update(self, transitions_dict):
        states = torch.tensor(transitions_dict['states'], dtype=torch.float).view(-1, self.n_states).to(self.device)
        actions = torch.tensor(transitions_dict['actions'], dtype=torch.float).view(-1, self.n_actions).to(self.device)
        rewards = torch.tensor(transitions_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transitions_dict['next_states'], dtype=torch.float).view(-1, self.n_states).to(self.device)
        dones = torch.tensor(transitions_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        #利用策略目标网络预测下一个状态的动作值
        next_actions = self.actor_target(next_states)

        #利用评价目标网络计算下一个状态动作对的价值，
        #得出当前时刻的动作价值的目标值
        next_q_values = self.critic_target(next_states, next_actions)
        q_targets_value = rewards + self.gamma * next_q_values * (1 - dones)

        #当前时刻动作价值的预测值
        q_values = self.critic(states, actions)

        # 均方差损失函数
        critic_loss = torch.mean(F.mse_loss(q_targets_value - q_values))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        ################

        #当前状态计算的动作值
        actions = self.actor(states)
        #计算当前状态动作对的价值
        score = self.critic(states, actions)

        actor_loss = -torch.mean(score)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #软更新策略网络和评价网络
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

