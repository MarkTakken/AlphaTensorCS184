## Implements the TensorGame logic

import torch

class TensorGame:
    def __init__(self, state, max_time, time=0):  # state is S x S x S integer tensor
        self.state = state
        self.max_time = max_time
        self.time = time
    
    def play(self, action):  # action is 3 x S tensor
        newstate = self.state - torch.einsum('i,j,k -> ijk', action[0], action[1], action[2])
        return (TensorGame(newstate, self.max_time, self.time+1), -1.0)

    def done(self):
        return self.time == self.max_time or torch.all(self.state == 0)
    
    def terminal_reward(self):
        return 0.0   # NOT YET IMPLEMENTED
    
    def nn_canonical(self):
        return self.state[None]
    
    def to_string(self, action=None):
        if action == None:
            return str(self.state)
        else:
            return str((self.state, action))