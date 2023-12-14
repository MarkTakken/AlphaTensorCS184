## Self-play is meant to control the self-play loop
from architecture import *
from mcts import *
from utilities import *
from tensorgame import *
import torch
import numpy as np
from tqdm import tqdm

canonical = torch.zeros(4, 4, 4)
canonical[0, 0, 0] = 1
canonical[0, 1, 1] = 1
canonical[1, 2, 0] = 1
canonical[1, 3, 1] = 1
canonical[2, 0, 2] = 1
canonical[2, 1, 3] = 1
canonical[3, 2, 3] = 1
canonical[3, 3, 3] = 1

def self_play(model, S: int, canonical, n_plays, num_samples = 16, num_sim = 16, identifier=1, max_actions = 16):
    model.eval()

    # Build a set of target tensors
    targets = [canonical] * n_plays
    bases_changes = [None] * n_plays
    for i, state in enumerate(targets):
        targets[i], cob = apply_COB(state, S, torch.tensor([-1, 0, 1]), torch.tensor([.05, .9, .05]))
        bases_changes[i] = cob
    # Play the game using MCTS and model for each tensor

    successful_trajectories = [] # Tuples of (Initial State, cob, [Actions], Final State)
    SAR_pairs = [] # Tuples of (State, Action, Reward)

    for i, target in tqdm(enumerate(targets)):
        ## Need to expand this
        root = TensorGame(target, num_samples, num_sim)
        mcts = MCTS(root, model)

        reward = 0
        states = []
        actions = []
        for i in range(max_actions):
            states.append(root.state)
            mcts.search(num_samples, num_sim)
            r, action = mcts.search_and_play(num_samples, num_sim)
            actions.append(action)
            reward += r

            if root.done():
                break
            if i == max_actions - 1:
                reward += root.terminal_reward()
        
        final_state = root.state
        
        SAR = [(state, action, reward * (i + 1) / len(actions)) for i, (state, action) in enumerate(zip(states, actions))]

        SAR_pairs += SAR

        if final_state == torch.from_numpy(np.zeros((S, S, S))):
            successful_trajectories.append((target, bases_changes[i], actions, final_state))
    
    torch.save(successful_trajectories, f"data/successful_trajectories_{identifier}.pt")
    torch.save(SAR_pairs, f"data/SAR_pairs_{identifier}.pt")
