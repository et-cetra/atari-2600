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
    Class references code from Lapan, M. <https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/02_dqn_pong.py>
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

        # Take random action with probability epsilon
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        # Otherwise act based on neural network
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

        # If we've finished the episode, return the total reward for this episode.
        # Otherwise return None
        episodeFinalReward = None
        if done:
            episodeFinalReward = self.totalReward
            self._reset()
        return episodeFinalReward

def huberLoss(loss):
    return 0.5 * loss ** 2 if abs(loss) < 1.0 else abs(loss) - 0.5

def calc_loss(batch, batchWeights, net, tgtNet, gamma, device):
    '''
    Args:
        net     - main neural network we are training. Used to calculate gradients.
        tgt_net - training-delayed copy of our neural net. Used to calculate values of state.
                  (won't affect the calculation of gradients)
    Function references code from Lapan, M. <https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/02_dqn_pong.py>
    '''
    states, actions, rewards, dones, nextStates = batch

    # Convert all these arrays to tensors/vectors
    states = torch.tensor(states).to(device)
    nextStates = torch.tensor(nextStates).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    doneMask = torch.BoolTensor(dones).to(device)

    # For priority experience buffer
    batchWeightsT = torch.tensor(batchWeights).to(device)

    # The following code attempts to exploit the GPU to make the code run faster 

    # Pass the states values of the batch to the neural net, and extract the specific Q values (state action values) for the taken actions using gather()
    temp = actions.unsqueeze(-1).long()
    stateActionValues= net(states).gather(1, temp).squeeze(-1)

    # Implement double DQN:
    # Select the best action to take in the next state using our MAIN network,
    # But take the values corresponding to this action from the TARGET network
    nextStateActions = net(nextStates).max(1)[1]
    nextStateValues = tgtNet(nextStates).gather(1, nextStateActions.unsqueeze(-1)).squeeze(-1)

    # If this was the end of the episode, obviously the value of future states should be 0
    nextStateValues[doneMask] = 0.0

    # Detach to stop gradients flowing into neural network
    nextStateValues = nextStateValues.detach()

    # Calculate Bellman approximation
    expectedStateActionValues = nextStateValues * gamma + rewards

    # We use Huber loss as a loss function to avoid gradient exploding (as it clips gradients between -1 and 1)
    # For priority experience buffer, we need to make our own Huber loss
    # Credit to https://stackoverflow.com/questions/60252902/implementing-smoothl1loss-for-specific-case
    # for this custom huber loss
    errors = torch.abs(stateActionValues - expectedStateActionValues)
    mask = errors < 1
    losses = (batchWeightsT * (0.5 * mask * (errors) ** 2)) + batchWeightsT * (~mask * (errors - 0.5))

    # We add a small value to every loss to handle the situation where the loss value is 0, which will lead to 0 priority
    return losses.mean(), losses + 1e-5


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

class PriorityExperienceBuffer:
    def __init__(self, capacity, alphaPriority):
        self.alphaPriority = alphaPriority
        self.capacity = capacity
        # Buffer should be a circular buffer of experiences
        self.buffer = np.empty((capacity, ), dtype=object)
        self.bufferSize = 0
        # Priorities will be a circular buffer of priorities (floats)
        # The element in each index is the priority of the element at the index in buffer
        self.priorities = np.zeros((capacity, ), dtype=np.float32)
        self.currentIndex = 0

    def __len__(self):
        return self.bufferSize

    def append(self, experience):
        if (self.bufferSize == 0):
            maxPriority = 1
        else:
            # Find the max value in the priority
            maxPriority = self.priorities.max()

        self.buffer[self.currentIndex] = experience
        self.priorities[self.currentIndex] = maxPriority
        # Circular buffer
        self.currentIndex = (self.currentIndex + 1) % self.capacity
        if (self.bufferSize < self.capacity):
            self.bufferSize += 1

    def sample(self, batchSize, beta):
        if (self.bufferSize == self.capacity):
            # Buffer is full, thus the whole array of priorities contain valid values
            priorities = self.priorities
        else:
            # Buffer not full, thus any element after currentIndex is not a valid value
            priorities = self.priorities[:self.currentIndex]

        # Do the calculations in the paper
        probabilities = priorities ** self.alphaPriority
        probabilities /= probabilities.sum()

        # Choose random experiences based on the probabilities
        indices = np.random.choice(self.bufferSize, batchSize, p=probabilities)
        states, actions, rewards, dones, nextStates = zip(*[self.buffer[i] for i in indices])

        # Calculate the weights
        weights = (self.bufferSize * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.bool), np.array(nextStates)), indices, weights

    def update_priorities(self, batchIndices, batchPriorities):
        for i, priority in zip(batchIndices, batchPriorities):
            self.priorities[i] = priority
