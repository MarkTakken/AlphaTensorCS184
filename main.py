## Main is meant to control the overall running of the program, switching between self-play and training

import torch
from architecture import *
from mcts import *
from training import *
from selfplay import *
from gc import collect
import math

def main(iterations = 10, S = 4, model_path = None, num_sim = 16, 
         max_actions = 16, n_plays = 1024, max_pregen = 500000, max_selfplay = 500000, epochs_per_iteration = 5, 
         batch_size = 1024, lr = 0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Either initialize or load the model
    for i in range(iterations):
        print(f"Iteration {i}: Beginning")
        print(f"Device: {device}")
        model = AlphaTensor184(s = 4, c = 48, d = 32, elmnt_range=(-2, 2), N_policy_features=32, N_policy_heads=8, Nsteps=4, Nsamples=24, torso_iterations=4)
        model.to(device)

        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print(f"Total memory: {t}")
        print(f"Reserved memory: {r}")
        print(f"Allocated memory: {a}")


        pytorch_total_params = sum(p.numel() for p in model.parameters())

        print(pytorch_total_params)

        print(f"Iteration {i}: Loading Data")

        if i == 0 and model_path != None:
            model.load_state_dict(torch.load(model_path))
        elif i > 0:
            model.load_state_dict(torch.load(f'models/model_{i-1}.pt'))
        model.train()
        if i % 3 == 0:
            pretrained = "data/SAR_pairs_4_100000_1.pt"
        elif i % 3 == 1:
            pretrained = "data/SAR_pairs_4_100000_2.pt"
        elif i % 3 == 3:
            pretrained = "data/SAR_pairs_4_100000_3.pt"

        selfplay_files = [f"data/SAR_pairs_sp_{i}.pt" for i in range(i)]
        dataset = ActionDataset([pretrained], max_pregen, max_selfplay, selfplay_files=selfplay_files, selfplay_multiplier=max(10-i, 4))

        print(f"Iteration {i}: Training")
        #Train
        train(model, dataset, epochs_per_iteration, batch_size = batch_size, lr=(lr / math.log(i + math.e)), device = device)
        torch.save(model.state_dict(), f'models/model_{i}.pt')

        model.eval()
        # Self-play
        print(f"Iteration {i}: Self-Playin")
        successes, avg_reward = self_play(model, S, canonical, n_plays = n_plays, num_sim = num_sim, identifier=i, max_actions = max_actions)

        print(f"Successful trajectories: {len(successes)}")
        print(f"Avg Reward: {avg_reward}")

        torch.cuda.empty_cache()
        collect()

if __name__ == "__main__":
    main()
