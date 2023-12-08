## Implements the MCTS using a provided model, returns a set of trajectories // tensors

import numpy as np

class MCTS:
    def __init__(self, root, nn, cpuct=1.0):
        self.root = root
        self.nn = nn
        self.cpuct = cpuct
        self.A = dict()         # state.to_string() -> sampled actions and probabilities
        self.N = dict()         # state.to_string() -> visit count
        self.Nsa = dict()       # state.to_string(action) -> visit count
        self.Q = dict()         # state.to_string() -> state value (value func)
        self.Qsa = dict()       # state.to_string() -> action value (Q func)
    
    def search(self, state):
        if state.done():
            return state.terminal_reward()
        
        key = state.to_string()
        nn_state = state.nn_canonical()
        
        # First time visiting state.  Simply collect value and backprop
        if not(key in self.N):
            self.N[key] = 1
            self.Q[key] = self.nn(nn_state)
            return self.Q[key]
        
        # Second time visiting state.  Expand the tree with the sampled states.
        if not(key in self.A):
            self.A[key] = self.nn.sample_actions(nn_state)
            for a in self.A[key].actions:
                keysa = state.to_string(a)
                self.Nsa[keysa] = 0
                self.Qsa[keysa] = float('inf')   # To ensure that we start by choosing each action once
        
        # Choose action with highest upper confidence bound
        best_u = -float('inf')
        best_a = None
        for (a,p) in zip(*self.A[key]):
            keysa = state.to_string()
            U = self.Qsa[keysa] + self.cpuct * p * np.log(self.N[key])/np.sqrt(self.Nsa[keysa]+1)
            if U > best_u:
                best_u = U
                best_a = a
        
        # Play the chosen action, perform a search on the new state and update the values and visit counts accordingly
        # Return the value of the current state up the branch
        newstate, reward = state.play(best_a)
        new_val = self.search(newstate)
        self.Q[key] = (self.Q[key]*self.N[key] + new_val + reward)/(self.N[key]+1)
        self.N[key] += 1
        keysa = state.to_string(best_a)
        self.Qsa[keysa] = (self.Qsa[keysa]*self.Nsa[keysa] + new_val + reward)/(self.Nsa[keysa]+1)
        self.Nsa[keysa] += 1
        return new_val + reward