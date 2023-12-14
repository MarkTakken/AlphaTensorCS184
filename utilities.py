import torch
from torch.utils.data import Dataset
import itertools

class Tokenizer():
    def __init__(self, range: tuple[int, int] = (-2, 2)) -> None:
        self.low, self.high = range
    
    def tokenize(self, tensor: torch.Tensor) -> torch.Tensor:
        dp = (tensor - self.low) * torch.Tensor([[1, (self.high-self.low) + 1, (self.high-self.low + 1)**2]])
        return torch.sum(dp, axis=1).int()

    def detokenize(self, token: torch.Tensor) -> torch.Tensor:
        a = torch.remainder(token, (self.high-self.low) + 1)
        b = torch.remainder(token - a, (self.high-self.low + 1)**2) // (self.high-self.low + 1)
        c = (token - a - b) // (self.high-self.low + 1)**2
        return (torch.column_stack((a, b, c)) + self.low).int()
    
    def batch_detokenize(self, token: torch.Tensor) -> torch.Tensor:
        a = torch.remainder(token, (self.high-self.low) + 1)
        b = torch.remainder(token - a, (self.high-self.low + 1)**2) // (self.high-self.low + 1)
        c = (token - a - b) // (self.high-self.low + 1)**2
        return (torch.stack((a,b,c), axis=2) + self.low).int()
    

class ActionDataset(Dataset):
    def __init__(self, pregen_files, max_pregen, max_selfplay, selfplay_files = None):
        self.max_pregen = max_pregen
        self.max_selfplay = max_selfplay
        l = [torch.load(file) for file in pregen_files]
        self.pregen_actions = list(itertools.chain.from_iterable(l))[:self.max_pregen]
        self.selfplay_actions = []
        if selfplay_files != None:
            l = [torch.load(file) for file in selfplay_files]
            self.selfplay_actions = list(itertools.chain.from_iterable(l))[:self.max_selfplay]

    def __len__(self):
        return len(self.pregen_actions) + len(self.selfplay_actions)

    def __getitem__(self, idx):
        if idx < self.max_pregen:
            return self.pregen_actions[idx]
        else:
            return self.selfplay_actions[idx - self.max_pregen]
        
    def add_selfplay_actions(self, actions):
        self.selfplay_actions = self.selfplay_actions + actions
        if len(self.selfplay_actions) > self.max_selfplay:
            self.selfplay_actions = self.selfplay_actions[len(self.selfplay_actions) - self.max_selfplay:]

def change_of_basis(S, cob_entries, cob_probs):
    P = torch.zeros(S, S, dtype=int)
    L = torch.zeros(S, S, dtype=int)
    diag_entries = torch.tensor([-1,1])
    diag_unif = torch.tensor([0.5, 0.5])
    diag_range = torch.arange(S)
    diag_elmnts = diag_entries[torch.multinomial(diag_unif, num_samples=2*S, replacement=True)]
    P[diag_range, diag_range] = diag_elmnts[:S]
    L[diag_range, diag_range] = diag_elmnts[S:]
    cob_elmnts = cob_entries[torch.multinomial(cob_probs, num_samples=S*S, replacement=True)].reshape(S,S)
    P += torch.triu(cob_elmnts, diagonal=1)
    L += torch.tril(cob_elmnts, diagonal=-1)
    return P @ L

def apply_COB(state, S, cob_entries, cob_probs):
    M = change_of_basis(S, cob_entries, cob_probs)
    state = torch.einsum('ijk, ia -> ajk', state, M.float())
    state = torch.einsum('ijk, ja -> iak', state, M.float())
    state = torch.einsum('ijk, ka -> ija', state, M.float())
    return state, M