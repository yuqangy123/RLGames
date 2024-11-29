
'''

'''
import numpy as np
import random


# class ReplayBuffer:    
#     def __init__(self, capacity):        
#         self.capacity = capacity        
#         self.buffer = []        
#         self.position = 0    
#     def push(self, state, action, reward, next_state, done):        
#         if len(self.buffer) < self.capacity:            
#             self.buffer.append(None)        
#         self.buffer[self.position] = (state, action, reward, next_state, done)        
#         self.position = (self.position + 1) % self.capacity    
    
#     def sample(self, batch_size):        
#         batch = random.sample(self.buffer, batch_size)
#         state, action, reward, next_state, done = zip(*batch)
#         return state, action, reward, next_state, done









# action_list = [np.zeros((7, 7)) for i in range(37)]
# #print(np.array(action_list).shape)
# print(len(action_list))
# action_list[-1][0, 6] = 1
# print('\n\n\n\n\n\n\n\n\n\n\n\n')
# #print(action_list)

# for i in range(6):
#     for j in range(6):
#         if i * 6 + j >= len(action_list):
#             break
#         action_list[i * 6 + j][:i + 1, j] = 1
#         print(f"[{i * 6 + j}][:{i + 1, j}]")
#         print(action_list[i * 6 + j])
#         x=0

# action_list = [mat[:6, :] for mat in action_list]
# for i in range(len(action_list)):
#     print(f"\r\n: {action_list[i]}")



arr = np.arange(0, 24, 1).reshape(2,3,4)
print(arr.shape)
print(arr)
print(arr[1,...])