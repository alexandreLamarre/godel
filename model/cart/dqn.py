'''
## Reference

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

## Overview 

For this example, 
our environment is deterministic. so all equations presented here are also formulated 
deterministically for the sake of simplicity.

In the reinforcement learning literature, they would also contain expectations over stochastic 
transitions in the environment.


Our aim will be to train a policy that tries to maximize, cumulatibe reward :

R_t0 = Σ_{t=0}^{inf} γ^(t-t0) * R_{t} where R_t0 is also known as the return.

The discount factor γ is a parameter that controls the relative importance of future rewards and
should be a constant between 0 and 1, to ensure that the sum converges.


The Main idea behind Q-learning is that if we hajd a function Q* : State X Action -> |R  that could
tell use what our return would be, if we were to take action in a given state, then we could
construct a policy that maximizes our rewards.

pi*(s) = argmax Q*(s,a)

However, we don't know everything about the world so we don't have access to Q*.
Since Neural Networks are universal function approximators, we can simpy 
create one and train it to resemble Q*.

For ou training update rule, we'll use the fact that every Q function for some policy obeys
the Bellman equation : 

https://en.wikipedia.org/wiki/Bellman_equation

The difference between the two sides of the equality is known as the temporal difference error.

To minimize this error, we will use the `HuberLoss`.


The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large 
- this makes it more robust to outliers when the estimates of Q are very noisy

Our Model will be a convoluational neural netowkr that takes in the difference between the current and
previous screen patches.
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
