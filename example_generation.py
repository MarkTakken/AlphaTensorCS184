## Creates the number of samples for the pre-computed examples

import sys
import random
import utilities
import numpy as np
import torch 
from tqdm import tqdm
## Should also define classes to handle data

## Convetion format for a trajectory
## Training (State, Action, Reward)
## Examples (State, Actions [], Reward)
#### instead [(State, Action, Reward)]

## MCTS (State) -> (Action)
## Self-play Null -> [(State, Action, Reward)]

## State is a tensor that is S x S x S
## Action is a list of length S, with each element being a token being a integer. 
## Reward is simply float

def generate_sample_r1(S: int, vals: list[int], factor_dist: list[float]):
    ## Generates a SxSxS tensor of rank 1 with nonzero entries by multiplying three Sx1 tensors
    nonzero = False
    while not nonzero:
        t = np.random.choice(vals, size=(3, S), p=factor_dist)
        m = np.tensordot(np.tensordot(t[0, :], t[1, :], axes=0), t[2, :], axes=0)
        assert m.shape == (S, S, S)
        nonzero = np.any(m)
    return t, m
    

def main(S: int, r_limit: int, factor_dist: dict, N: int, seed: int = None):
    # The primary method of generation of synthetic trajectories
    ## Arguments:
    ## S as the dimension of the tensor
    ## r_limit as an int, which is the maximum number of actions in the trajectory
    ## factor_dist as a dict, which is the distribution of factors
    ## N as an int, which is the number of trajectories to generate
    ## seed as an int, which is the seed for the random number generator. If none, random
    if seed is not None:
        random.seed(seed)

    low, high = min(factor_dist.keys()), max(factor_dist.keys())
    vals = list(factor_dist.keys())
    dist = [factor_dist[i] for i in factor_dist.keys()]

    
    tokenizer = utilities.Tokenizer(range=(low, high))

    SAR_pairs = []
    
    for i in tqdm(range(N)):
        R = random.randint(1, r_limit)
        T = torch.zeros((S, S, S), dtype=torch.int)
        reward = 0
        for j in range(R):
            sample, m = generate_sample_r1(S, vals, dist)
            T += torch.from_numpy(m)
            tokens = tokenizer.tokenize(torch.from_numpy(sample.T))
            reward += -1
            SAR_pairs.append((T, tokens, reward))

    return SAR_pairs
            

if __name__ == "__main__":
    S = 4
    r_limit = 16
    factor_dist = {-2: .001, -1: .099, 0: .98 , 1: .099, 2: .001}
    N = 100000
    seed = 6
    SAR_pairs = main(S, r_limit, factor_dist, N, seed=seed)

    torch.save(SAR_pairs, f"data/SAR_pairs_{S}_{N}_{seed}.pt")