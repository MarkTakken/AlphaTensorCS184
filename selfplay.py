## Self-play is meant to control the self-play loop
from architecture import *
from mcts import *
from utilities import *
from tensorgame import *
import torch
import numpy as np
from tqdm import tqdm

# Probably more convenient for state to be a tensor of ints
#       so that we don't risk floating point inaccuracies when
#       checking whether the state equals zero
canonical = torch.zeros(4, 4, 4, dtype=torch.long)
canonical[0, 0, 0] = 1
canonical[0, 1, 1] = 1
canonical[1, 2, 0] = 1
canonical[1, 3, 1] = 1
canonical[2, 0, 2] = 1
canonical[2, 1, 3] = 1
canonical[3, 2, 3] = 1
canonical[3, 3, 3] = 1

def self_play(model, S: int, canonical, n_plays, num_samples = 16, num_sim = 16, identifier=1, max_actions = 12,
              cob_entries = torch.tensor([-1, 0, 1]), cob_probs = torch.tensor([.05, .9, .05]), device='cuda'):
    model.eval()

    # Build a set of target tensors
    targets = [canonical] * n_plays
    bases_changes = [None] * n_plays
    for i, state in enumerate(targets):
        cob = change_of_basis(S, cob_entries, cob_probs)
        targets[i] = apply_COB(state, cob)
        bases_changes[i] = cob
    # Play the game using MCTS and model for each tensor

    # Not including cob in successful_trajectories anymore b/c we can just
    #       immediately perform the reverse change of basis (see below)
    successful_trajectories = [] # Tuples of (Initial State, [Actions], Final State)
    SAR_pairs = [] # Tuples of (State, Action, Reward)

    total_reward = 0

    for i, target in tqdm(enumerate(targets)):
        ## Need to expand this
        # Avoding separate root = TensorGame(target, max_actions'')
        #     for clarity: We want to work with mcts.root, which is
        #     updated with mcts.search_and_play, rather than root.
        mcts = MCTS(TensorGame(target, max_actions), model, device=device)

        # Storing all rewards; see comment below
        rewards = []
        states = []
        actions = []
        for _ in range(max_actions):
            states.append(mcts.root.state)
            # search_and_play already calls search internally
            r, action = mcts.search_and_play(num_samples, num_sim)
            actions.append(action)
            rewards.append(r)

            if mcts.root.done():  # Already considers i == max_actions - 1
                break
        
        rewards[-1] += mcts.root.terminal_reward()
        
        # I think the value should be the sum of the suffix of the list of rewards
        #      rather than smearing the total reward equally over all actions.  For
        #      instance, I think the last station-action pair should have reward
        #      -1 + terminal rather than (-n + terminal)/n = -1 + terminal/n, where
        #      n = len(actions)
        SAR = []
        value = 0
        for (state, action, reward) in zip(reversed(states), reversed(actions), reversed(rewards)):
            value += reward
            # Maybe we could consider canonicalizing the action by sorting or smnth
            SAR.append((state, action, value))

        SAR_pairs += SAR

        total_reward += reward

        if mcts.root.is_zero():
            orig_actions = [action @ bases_changes[i] for action in actions]
            successful_trajectories.append((target, orig_actions))
    
    torch.save(successful_trajectories, f"data/successful_trajectories_{identifier}.pt")
    torch.save(SAR_pairs, f"data/SAR_pairs_sp_{identifier}.pt")

    return successful_trajectories, total_reward / n_plays
