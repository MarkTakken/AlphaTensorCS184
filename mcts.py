## Implements the MCTS using a provided model, returns a set of trajectories // tensors

import torch

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
    
    def single_search(self, state, num_samples):
        if state.done():
            return state.terminal_reward()
        
        key = state.to_string()
        nn_state = state.nn_canonical()
        
        # First time visiting state.  Collect value, expand tree and backprop
        if not(key in self.N):
            self.N[key] = 1
            self.Q[key], self.A[key] = self.nn(nn_state)  # Assumes nn returns detokenized actions
            for a in self.A[key].actions:
                keysa = state.to_string(a)
                self.Nsa[keysa] = 0
                self.Qsa[keysa] = 0.0
            return self.Q[key]
        
        # Choose action with highest upper confidence bound
        best_u = -float('inf')
        best_a = None
        for (a,p) in zip(*self.A[key]):
            keysa = state.to_string(a)
            # Intentional initial division by 0 to ensure each action is chosen once
            U = self.Qsa[keysa] + self.cpuct * p * torch.log(torch.tensor(self.N[key]+1))/torch.sqrt(torch.tensor(self.Nsa[keysa]))
            if U > best_u:
                best_u = U
                best_a = a
        
        # Play the chosen action, perform a search on the new state and update the values and visit counts accordingly
        # Return the value of the current state up the branch
        newstate, reward = state.play(best_a)
        new_val = self.single_search(newstate, num_samples)
        self.Q[key] = (self.Q[key]*self.N[key] + new_val + reward)/(self.N[key]+1)
        self.N[key] += 1
        keysa = state.to_string(best_a)
        self.Qsa[keysa] = (self.Qsa[keysa]*self.Nsa[keysa] + new_val + reward)/(self.Nsa[keysa]+1)
        self.Nsa[keysa] += 1
        return new_val + reward
    
    def search(self, num_samples, num_sim):
        for _ in range(num_sim):
            self.single_search(self.root, num_samples)
    
    # Actions are chosen proportionally to (visit count)^(1/temp)
    def get_action_probs(self, temp=1.0):
        key = self.root.to_string()
        if temp > 0:
            probs = torch.zeros(len(self.A[key].actions))
            s = 0
            for (i,a) in enumerate(self.A[key].actions):
                weight = self.Nsa[self.root.to_string(a)]**(1/temp)
                s += weight
                probs[i] = weight
            return probs/s
        
        # Handle temp = 0 case separately, which is just argmax
        else:
            best_i, best_n = None, 0
            for (i,a) in enumerate(self.A[key].actions):
                count = self.Nsa[self.root.to_string(a)]
                if count > best_n:
                    best_n = count
                    best_i = i
            probs = torch.zeros(len(self.A[key].actions))
            probs[best_i] = 1.0
            return probs
    
    def choose_move(self, temp=1.0):
        probs = self.get_action_probs(temp=temp)
        actions = self.A[self.root.to_string()].actions
        ind = torch.multinomial(probs, 1).item()
        return actions[ind]
    
    def search_and_play(self, num_samples, num_sim, temp=1.0):
        self.search(num_samples, num_sim)
        action = self.choose_move(temp=temp)
        self.root, reward = self.root.play(action)
        return reward