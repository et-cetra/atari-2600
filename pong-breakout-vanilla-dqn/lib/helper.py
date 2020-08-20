#
# Code assembled and tweaked by:
#   Joshua Derbe
#   Sean Luo
#   Mariya Shmalko
#   Arkie Owen
#
#
# Base algorithm outline created with heavy reference to
#   Lapan, M. (2018) Deep Reinforcement Learning Hands-On, Packt Publishing.
#

import collections
import numpy as np
import torch
import torch.nn as nn

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
class ExperienceBuffer:
    '''
    This is the buffer that will hold a given number of past observations and sample from them
    This is the method that tries to remove correlation in the environment steps to help SGD
    '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batchSize):
        # Choose a batchSize number of experiences at random
        indices = np.random.choice(len(self.buffer), batchSize, replace=False)
        states, actions, rewards, dones, nextStates = zip(*[self.buffer[i] for i in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.bool), np.array(nextStates)


class Agent:
    '''
    Class to interact with the environment
    Class references code from Lapan, M. <https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/02_dqn_pong.py>
    '''
    def __init__(self, env, expBuffer):
        self.env = env
        self.expBuffer = expBuffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.totalReward = 0.0

    def play_step(self, net, epsilon, device):

        # We either take a random action...
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        # ...or act based on our neural network.
        # As epsilon decreases, we are more likely to use the neural network over epsilon
        else:
            # Turn our state into a tensor
            state = np.array([self.state], copy=False)
            state = torch.tensor(state).to(device)
            # Calculate the q values of actions we can take given this state
            qValues = net(state).to(device)
            # The action we choose is the one with the highest q value
            _, action = torch.max(qValues, dim=1)
            action = int(action.item())

        # Once we have our action, we can act in our environment.
        newState, reward, done, _ = self.env.step(action)
        self.totalReward += reward

        # Experience is added to the buffer, state updated
        exp = Experience(self.state, action, reward, done, newState)
        self.expBuffer.append(exp)
        self.state = newState
        episodeFinalReward = None
        if done:
            # If we've finished the episode, return the total reward for this episode.
            # Otherwise return None
            episodeFinalReward = self.totalReward
            self._reset()
        return episodeFinalReward

def calc_loss(batch, net, tgtNet, gamma, device):
    '''
    Args:
        net     - main neural network we are training. Used to calculate gradients.
        tgt_net - training-delayed copy of our neural net. Used to calculate values of state.
                  (won't affect the calculation of gradients)

    '''
    states, actions, rewards, dones, nextStates = batch

    # Convert all these arrays to tensors/vectors
    states = torch.tensor(states).to(device)
    nextStates = torch.tensor(nextStates).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    doneMask = torch.ByteTensor(dones).to(device)

    # The following code is not very readable as it attempts to exploit the GPU to make the code run faster 

    # Pass the states values of the batch to the neural net, and extract the specific Q values (state action values) for the taken actions using gather()
    temp = actions.unsqueeze(-1).long()
    stateActionValues= net(states).gather(1, temp).squeeze(-1)

    # Get the values of the next states from our target network
    nextStateValues = tgtNet(nextStates).max(1)[0]

    # If this was the end of the episode, obviously the value of future states should be 0
    nextStateValues[doneMask] = 0.0

    # Detach to stop gradients flowing into neural network
    nextStateValues = nextStateValues.detach()

    # Calculate Bellman approximation
    expectedStateActionValues = nextStateValues * gamma + rewards

    # Use MSE Loss 
    return nn.MSELoss()(stateActionValues, expectedStateActionValues)


class RewardBuffer:
    '''
    RewardBuffer class
    Used to store the reward over a fixed number of past games to assess how well the network is learning
    '''
    def __init__(self,maxlength):
        self.buffer = collections.deque(maxlen=maxlength)
    def append(self,item):
        self.buffer.append(item)
    def mean(self):
        return np.mean(self .buffer)