## Creates the number of samples for the pre-computed examples

import sys
import random
import utilities

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

def main(S, r_limit, factor_dist, N, seed=None):
    if seed is not None:
        random.seed(seed)

    low = -1 * len(factor_dist) // 2
    high = len(factor_dist) // 2
    
    tokenizer = utilities.Tokenizer(range=(-2, 2))
    pass

if __name__ == "__main__":
    args = sys.argv[1:]
    main()