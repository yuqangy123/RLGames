
'''
https://wenku.csdn.net/column/216ws7yf42?spm=1055.2635.3001.10023.4
Q-Learning算法的核心在于更新Q值，以不断优化策略。Q值的更新通常遵循以下公式：

[ Q(s, a) = Q(s, a) + \alpha \cdot (R(s) + \gamma \cdot \max Q(s’, a’) - Q(s, a)) ]
其中，(\alpha)为学习率，(\gamma)为折扣因子


V(s)←V(s)+α*( R[t+1] + γ*V(s′) − V(s) )
其中 R[t+1]+γV(s′) 称为TD目标      δt = R[t+1] + γ*V(s′) − V (s) 称为TD偏差
优先级经验回放的核心思想是根据样本的TD误差来确定其优先级    高误差样本被更频繁地抽样    以提高训练的效率和稳定性。

Q表格的值将根据当前状态执行的动作、获得的奖励以及下一个状态的最大Q值进行更新 以不断优化策略。
'''
import numpy as np


# 定义迷宫环境 5X5
env_s = 5
env = np.array([
    [0, 0,  0, -1, 0],
    [0, -1, 0, -1, 0],
    [0,  0, 0,  0, 0],
    [-1, 0, -1, 0, 0],
    [0,  0, 0,  0, 1]
])

#奖励: 到达终点奖励为100，撞到障碍物奖励为-10，其他情况奖励为-1
# 在Q-Learning算法中，奖励的设置主要涉及以下几个方面：
# 奖励函数的设计：奖励函数应该根据特定问题的目标来设计，以确保Agent能够学习到最优的策略。奖励函数应该具有良好的目标导向性，即奖励机制应该能够引导Agent向最终目标迈进1。
# 奖励的更新：Q-Learning算法通过执行动作并获得奖励来更新Q表。在每个状态s下，根据动作a执行后所获得的奖励r，以及下一个状态s'，计算Q值的更新，即Q(s, a) = R(s, a) + γmax{Q(s', a')}，其中γ是学习参数，R是奖励机制4。
# 奖励矩阵：在Q-Learning中，奖励矩阵是一个表格，其中每一行代表一个状态（State），每一列代表一个动作（Action）。矩阵中的每个元素Rs,a表示在特定状态s下采取特定动作a的奖励3。
# 奖励的衰减：Q-Learning算法在更新Q表时，会根据状态之间的距离（即状态转移矩阵中的元素）来衰减奖励，离当前状态越远的奖励衰减的越严重2。
# 奖励的动态变化：在Q-Learning算法中，虽然奖励函数本身不发生变化，但Agent根据不同动作所接收到的奖励会动态变化，这有助于Agent在不断变化的环境中学习并优化其策略7。


# 动作值函数：定义Q表格(Q表格就是要学习的Agent的大脑)
Q = np.zeros((25, 4)) # 25个状态，4个动作（0上，1下，2左，3右）

def step(state, action):
    y,x = state // env_s, state % env_s
    isout = True
    if 0 == action:
        if y - 1 >= 0:
            y = y - 1
            isout = False
    elif 1 == action:
        if y + 1 < env_s:
            y = y + 1
            isout = False
    elif 2 == action:
        if x - 1 >= 0:
            x = x - 1
            isout = False
    elif 3 == action:
        if x + 1 < env_s:
            x = x + 1
            isout =False

    if isout:
        return state, -10, False

    if env[y, x] == 1:
        return y * env_s + x, 100, True
    elif env[y, x] == -1:
        return state, -10, False
    else:
        return y * env_s + x, -1, False
    
# Q-Learning算法
num_episodes = 100
learning_rate = 0.9
discount_factor = 0.99
for episode in range(num_episodes):
    state = 0
    done = False
    while not done:
        action = np.argmax(Q[state])
        new_state, reward, done = step(state, action)
        
        
        # 更新Q值
        Q[state, action] = Q[state, action] + learning_rate * (reward + 
            discount_factor * np.max(Q[new_state]) - Q[state, action])
        state = new_state

#Q矩阵规范化
Q = Q / np.max(Q)

print('\r\nQ矩阵:')
for row in range(len(Q)):
    print(row+1, Q[row])

