## Main is meant to control the overall running of the program, switching between self-play and training

import torch
from architecture import *
from mcts import *
from training import *
from selfplay import *

def main(iterations = 10, S = 4, model_path = None, num_samples = 16, num_sim = 16, 
         max_actions = 16, n_plays = 50000, max_pregen = 500000, max_selfplay = 500000, epochs_per_iteration = 5, 
         batch_size = 1024, lr = 0.01, device = 'cuda'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Either initialize or load the model
    model = AlphaTensor184(s = 4, c = 64, d = 48, elmnt_range=(-2, 2), Nsteps=4, Nsamples=32, torso_iterations=5)
    model.to(device)

    if model_path != None:
        model.load_state_dict(torch.load(model_path))
    for i in range(iterations):
        model.train()
        if i % 3 == 0:
            pretrained = "data/SAR_pairs_4_100000_1.pt"
        elif i % 3 == 1:
            pretrained = "data/SAR_pairs_4_100000_2.pt"
        elif i % 3 == 3:
            pretrained = "data/SAR_pairs_4_100000_3.pt"

        selfplay_files = [f"data/SAR_pairs_sp_{i}.pt" for i in range(i)]
        dataset = ActionDataset([pretrained], max_pregen, max_selfplay, selfplay_files=selfplay_files)
        #Train
        train(model, dataset, epochs_per_iteration, batch_size = batch_size, lr=lr, device = device)
        torch.save(model.state_dict(), f'./model_{i}.pt')

        model.eval()
        # Self-play
        successes, avg_reward = self_play(model, S, canonical, n_plays = n_plays, num_samples = num_samples, num_sim = num_sim, identifier=i, max_actions = max_actions)

        print(f"Successful trajectories: {len(successes)}")
        print(f"Avg Reward: {avg_reward}")

if __name__ == "__main__":
    main()
