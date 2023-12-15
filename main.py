## Main is meant to control the overall running of the program, switching between self-play and training

import torch
from architecture import *
from mcts import *
from training import *
from selfplay import *
from gc import collect
import math

def train_selfplay_loop(iterations = 10, S = 4, model_path = None, num_sim = 2, 
         max_actions = 16, n_plays = 1024, max_pregen = 500000, max_selfplay = 500000, epochs_per_iteration = 1, 
         batch_size = 1024, lr = 0.02):
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
        dataset = ActionDataset([pretrained], max_pregen, max_selfplay, selfplay_files=selfplay_files, selfplay_multiplier=max(11-i, 4))

        print(f"Iteration {i}: Training")
        #Train
        train(model, dataset, epochs_per_iteration, batch_size = batch_size, lr=(lr / math.log(i + math.e)), device = device)
        torch.save(model.state_dict(), f'models/model_{i}.pt')

        model.eval()
        # Self-play
        print(f"Iteration {i}: Self-Playing")
        with torch.no_grad():
            successes, avg_reward = self_play(model, S, n_plays = n_plays, num_sim = num_sim, identifier=i, max_actions = max_actions)

        print(f"Successful trajectories: {len(successes)}")
        print(f"Avg Reward: {avg_reward}")

        torch.cuda.empty_cache()
        collect()


def im_based_training(S = 4, epochs = (10, 10), model_path=None, num_sim = 50, n_plays = 20, lr = 0.001, id = 3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Either initialize or load the model
    model = AlphaTensor184(s = 4, c = 48, d = 48, elmnt_range=(-2, 2), N_policy_features=48, N_policy_heads=12, Nsteps=4, Nsamples=24, torso_iterations=4)
    model.to(device)

    if model_path != None:
        model.load_state_dict(torch.load(model_path))
    model.train()

    # Train
    meta_iterations, iterations = epochs

    datafiles = [f"data/SAR_pairs_4_100000_{i}.pt" for i in range(1, 2)]
    dataset = ActionDataset(datafiles, 750000, 100000)
    all_losses = []
    pol_losses = []
    val_losses = []
    for i in tqdm(range(meta_iterations)):
        losses, pl, vl= train(model, dataset, iterations, batch_size = 1024, lr=(lr / math.log(i * iterations + math.e)), device = device, val_weight=.05)
        torch.save(model.state_dict(), f'models/model_{id}_{i}.pt')
        all_losses += losses
        pol_losses += pl
        val_losses += vl
        print(f"All Losses: {all_losses}")
        print(f"Policy Losses: {pol_losses}")
        print(f"Value Losses: {val_losses}")

    print(f"Beginning Evaluation")
    model.eval()
    # Self-play
    with torch.no_grad():
        successes, avg_reward = self_play(model, S, n_plays = n_plays, num_sim = num_sim, identifier=1)

    print(f"Successful trajectories: {len(successes)}")
    print(f"Avg Reward: {avg_reward}")

    torch.save(model.state_dict(), "models/model_{id}_f.pt")

if __name__ == "__main__":
    im_based_training(model_path=None, id=31)
