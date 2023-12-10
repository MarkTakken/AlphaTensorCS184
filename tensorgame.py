## Implements the TensorGame logic

import torch

class TensorGame:
    def __init__(self, state, max_time):  # state is S x S x S integer tensor
        self.state = state
        self.max_time = max_time
        self.time = 0
    
    def play(self, action):  # action is 3 x S tensor
        newstate = torch.einsum('i,j,k -> ijk', action[0], action[1], action[2])
        self.time += 1
        return (newstate, -1.0)

    def done(self):
        return self.time == self.max_time or np.all(self.state == 0)
    
    def terminal_reward(self):
        raise NotImplementedError  # Still not sure how to best estimate/bound tensor rank
    
    def nn_canonical(self):
        return self.state[None]
    
    def to_string(self, action=None):
        if action == None:
            return str(self.state)
        else:
            return str((self.state, action))