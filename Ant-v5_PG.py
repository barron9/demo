import gymnasium as gym
import numpy as np
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable
from torch.distributions import Categorical
#env.seed(1); 
torch.manual_seed(1);

 
# parameter
learning_rate = 0.001
gamma = 0.95
num_episodes = 5
max_episode_steps = 800
# Track rewards per episode & training
episode_rewards = []
train = True 

env = gym.make('Ant-v5', render_mode="rgb_array",max_episode_steps=max_episode_steps)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)  # Mean of the distribution (mu)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)  # Log std (for numerical stability)
        self.gamma = gamma
        self.episode_gradients = []
        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor()) 
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)  # Mean of the action distribution
        log_std = self.fc_log_std(x)  # Log standard deviation
        std = torch.exp(log_std)  # Convert log std to actual standard deviation
        return mu, std

 
policy = PolicyNetwork(105, 8, 128)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

def select_action(policy_net, state):
    # Forward pass to get the mean and standard deviation of the action distribution
    mu, std = policy_net(Variable(torch.Tensor(state)))

    # Create the Normal distribution (for continuous action space)
    dist = Normal(mu, std)

    # Sample actions from the distribution
    action = dist.sample()  # Shape will be [batch_size, action_dim], in this case [1, 8]

    # Get the log probability of the sampled action
    log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # Sum over all action dimensions

 # Add log probability of our chosen action to our history    
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history, log_prob])
    else:
        policy.policy_history = (log_prob)
    return action, log_prob

def update_policy():
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)
        
    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    #     # Check gradients
    # for name, param in policy.named_parameters():
    #     if param.grad is not None:
    #         print(f"Gradient for {name} - Min: {param.grad.min()}, Max: {param.grad.max()}, Mean: {param.grad.mean()}")
    #policy.fc1.weight.detach().cpu().numpy()
    
    # Gradient clipping: clip gradients before the optimizer step
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
       # Check gradients
    # for name, param in policy.named_parameters():
    #     if param.grad is not None:
    #         print(f"after Gradient for {name} - Min: {param.grad.min()}, Max: {param.grad.max()}, Mean: {param.grad.mean()}")

    #Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []
    #episode_gradients = []


def main(episodes):
    running_reward = 10
    for episode in range(episodes):
        state,_ = env.reset() # Reset environment and record the starting state
        done = False       
        t = False
        while not (t or done):
            action = select_action(policy, state)
            # Step through environment using chosen action
            state, reward, done, t, reward_ctrl = env.step(action[0].numpy())

            # Save reward
            policy.reward_episode.append(reward)
            if  np.abs(state[0]) < 0.35:
                print("episode {} reward:{}".format(episode,np.sum(policy.reward_episode)))
                break
            if t or done: 
                print("episode {} reward:{}".format(episode,np.sum(policy.reward_episode)))
                break

                
        update_policy()
    
   
episodes = 12000
main(episodes)
env.close()
#-------------------------


# # Load with NumPy
# # Q = np.load('qtable.npy')
# import pickle
# with open('data.pkl', 'wb') as file:
#     pickle.dump(policy, file)

# with open('data.pkl', 'rb') as file:
#     policy = pickle.load(file)
test_run = 100
env_test = gym.make('Ant-v5', render_mode="human",max_episode_steps=2000)
episode_rewards = []
for i in range(test_run):
    state,_ = env_test.reset() # Reset environment and record the starting state
    done = False       
    t = False
    while not (t or done):
        action = select_action(policy, state)
        # Step through environment using chosen action
        state, reward, done, t, reward_ctrl = env_test.step(action[0].numpy())
        if np.abs(state[0]) < 0.31:
            break
 

env_test.close()
