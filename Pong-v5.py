import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F
import pickle
resume = True
env = gym.make('ALE/Pong-v5',render_mode="human")
#env.seed(1); 
torch.manual_seed(1);
#print(env.unwrapped.ale.getAvailableDifficulties())
#env.env.game_difficulty = 3
#env.unwrapped.ale.setDifficulty(3)
#Hyperparameters
learning_rate = 0.01
gamma = 0.99
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        #self.conv4 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=40, stride=2, padding=1)
        self.fc = nn.Linear(40*40, 128, bias=True)   
   
        self.fc3 = nn.Linear(128, 6)  
        self.gamma = gamma
        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
         # Move layers to the appropriate device
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        # Pass through convolutional layer
        #x = self.conv4(x)
       # x = x.view(x.size(0), -1)  
        x = self.fc(x)
    
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

# class Policy(nn.Module):
#     def __init__(self):
#         super(Policy, self).__init__()
#         self.state_space =  80 # env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
#         self.action_space = env.action_space.n
#        # self.flatten = nn.Flatten()  # Flattens the input from (210, 160, 3) to a 1D vector
#         self.l1 = nn.Linear(80*80, 200, bias=True)
#         self.l2 = nn.Linear(200, self.action_space, bias=True)
#         self.gamma = gamma
#         # Episode policy and reward history 
#         self.policy_history = Variable(torch.Tensor())
#         self.reward_episode = []
#         # Overall reward and loss history
#         self.reward_history = []
#         self.loss_history = []
#          # Move layers to the appropriate device
#         self.to(device)

#     def forward(self, x):    
#         x = x.to(device)
#         model = torch.nn.Sequential(
#             self.l1,
#             ##nn.Dropout(p=0.6),
#             nn.ReLU(),
#             self.l2,
#             nn.Sigmoid()
#         )

#         return model(x)


policy = Policy()
if resume: policy = pickle.load(open('./save_policy_li.p', 'rb'))

#optimizer = optim.RMSprop(policy.parameters(), lr=learning_rate)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  
  I = I[::4,::4,0] # downsample by factor of 2
  #I = I[:,:,0] # first chanel
#  I = I[:,20:140] # crop
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  # Display the image
#   plt.imshow(I)  # 'gray' for grayscale image
#   plt.colorbar()  # Optional: Shows color bar to interpret pixel values
#   plt.show()
  return I.astype(float) #torch.tensor(I.astype(float))#.unsqueeze(0).unsqueeze(0).to(device)

def select_action(state):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state_tensor = torch.from_numpy(state).type(torch.FloatTensor)
   # state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)#.to(device)
    state_flattened = state_tensor.view(-1)  # Flatten the tensor to 1D

    state = policy(Variable(state_flattened))
    c = Categorical(state) 
    action = c.sample()
    
    # Add log probability of our chosen action to our history    
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history.to(device), c.log_prob(action).to(device).unsqueeze(0)]).to(device)
    else:
        policy.policy_history = (c.log_prob(action)).to(device)
    return action

def update_policy(episode_cnt):
    R = 0
    rewards = []
    discounted_r = np.zeros_like(policy.reward_episode)
    running_add = 0
    for t in reversed(range(0, len(policy.reward_episode))):
        if policy.reward_episode[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + policy.reward_episode[t]
        discounted_r[t] = running_add
    rewards = discounted_r
    # Scale rewards
    rewards = torch.FloatTensor(rewards).to(device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Calculate loss
    #squeaze
    loss = (torch.sum(torch.mul(policy.policy_history.squeeze().to(device), Variable(rewards).to(device)).mul(-1), -1)).to(device)
    loss.backward()
    if episode_cnt % 6 == 0:
        optimizer.step()
        optimizer.zero_grad()
    # Update network weights
  #  if episode_cnt % 10 == 0:
  #      optimizer.step()
        

    #Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor()).to(device)
    policy.reward_episode= []

def main():
    episode_number = 0
    state,_ = env.reset() 

    while True:  
        action = select_action(prepro(state))
        state, reward, done, info, _ = env.step(action.item())
        policy.reward_episode.append(reward)
        if done:
            episode_number += 1
            print('episode:{} result:{}'.format(episode_number,np.sum(policy.reward_episode)))
            update_policy(episode_number)
            state,_ = env.reset() 
            if episode_number % 10:
                import pickle
                pickle.dump(policy, open('./save_policy_li.p', 'wb'))

main()
