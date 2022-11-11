import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import resource


class Network(nn.Module):
    '''Architecture of Neural Network'''

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        # input connection to hidden layer (input size, hidden layer size)
        self.fc1 = nn.Linear(input_size, 30)
        # hidden layer connection to output layer (hidden layer size, output layer size)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        # activating the first layer of connection (input -> hidden) with current state inputs
        x = F.relu(self.fc1(state))
        # getting q_values by already connection X -> connection fc2: output
        q_values = self.fc2(x)
        return q_values


class ReplayMemory(object):
    '''Architecture of Long Term Memory Replay'''

    def __init__(self, capacity):
        sys.setrecursionlimit(10**6)

        self.capacity = capacity  # number of transitions to store in memory
        self.memory = []

    def push(self, transition):
        '''
        transition: (last_state, new_state, last_action, last_reward)
        Push new transition into the memory.
        If memory is full, free up space by deleting first transition.
        '''

        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            print('Memory is full, deleting first transition')
            del self.memory[0]

    def sample(self, batch_size):
        '''
        batch_size: number of random sample transitions from memory.
        Returns random sample of transitions with size: batch_size from memory mapped with torch.
        '''

        # we want to have sample with this format: [(last_state1, last_state2,...), (last_action1,...), (last_reward1...)]
        # instead of [(last_state1, last_action1, last_reward1), (last_state2, last_action2, last_reward2), ...]
        samples = zip(*random.sample(self.memory, batch_size))

        # get list of batches (state_batch, action_batch, reward_batch) well aligned
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn():
    '''Deep Q Learning'''

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        # we want to learn looking at last 100.000 transitions
        self.memory = ReplayMemory(100000)
        # choosing the final action based on learning. Learning Rate slow to learn deeper
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # state is the group name of inputs. Inputs are: left_sensor, right_sensor, front_sensor, orientation, -orientation
        self.last_state = torch.Tensor(input_size).unsqueeze(0)  # input state

        # actions are: 0,1,2 and we will map them to orientation 0,20,-20(deg). 0=0, 1=20deg, 2=-20deg
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        '''
        state: Network output is action, which depends on input state (3 signals, 2 orientations).
        Will get 3 possible actions, and using Softmax we will choose the best action.
        '''

        # choose using softmax, but to choose the best action, we must try all actions.
        # we must have Q actions for each action. Proababilities for each Q value
        # softmax will map the highest probability to highest Q value
        '''
        Example: q_values=[1,2,3], probs=[0.04,0.11,0.85]
                 softmax([1,2,3]) = [0.04, 0.11, 0.85] / softmax([1,2,3]*3) = [0, 0.02, 0.98] #increasing T will increase probability of highest and decrease the lowest ones
                 Increasing T, now we are sure the right action to play is 3
        '''
        # if we want to deactive brain of the car we can put T=0
        # T=7: Temperature parameter, small->insect, increasing->car. Higher the temperature, higher probability for winning q value
        probs = F.softmax(self.model(Variable(state, volatile=True))*0)
        action = probs.multinomial(3)  # choose action based on probability
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        '''
        These are batches of states, rewards, and actions that we getting from ReplayMemory.sample()
        '''

        batch_action = batch_action.type(torch.int64)  # convert to int64
        # we will get outputs for all actions 1,2,3 without gather(). But we want only for actions that network decided to play each time
        outputs = self.model(batch_state).gather(
            1, batch_action.unsqueeze(1)).squeeze(1)  # vector of outputs
        # to compute the loss. We must use max(Q_value) in the formula
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward  # target that we want to achieve
        td_loss = F.smooth_l1_loss(outputs, target)

        # we have loss, and we want to backpropagate & update the weights
        self.optimizer.zero_grad()  # reinitialize optimizer in each iteration of the loop
        td_loss.backward(retain_varaibles=True)
        self.optimizer.step()  # Backpropagate the loss and update the weights

    def update(self, reward, new_signal):
        '''
        reward: Last reward
        signal: New signal. (3 signals, 2 orientations)
        '''

        new_state = torch.Tensor(new_signal).float().unsqueeze(0)

        self.memory.push((self.last_state, new_state, torch.LongTensor(
            [int(self.last_action)]), torch.Tensor([self.last_reward])))

        # we updated last transition in our memory, so we are in a new state now and we must play a new action!
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(
                100)
            self.learn(batch_state, batch_next_state,
                       batch_reward, batch_action)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return action

    def score(self):
        # +1 is save for division by 0
        return sum(self.reward_window) / (len(self.reward_window)+1.)

    def save(self):
        '''
        Save network
        Save optimizer. We want to save the last trained weights
        '''

        torch.save({'state_dict': self.model.state_dict(),
                   'optimizer': self.optimizer.state_dict}, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('=> loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done!')
        else:
            print('no checkpoint found...')
