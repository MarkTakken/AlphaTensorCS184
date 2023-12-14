from architecture import *
from mcts import *
from utilities import *
from tensorgame import *
from selfplay import *
import torch
import numpy as np

alphaTensor184 = AlphaTensor184(s = 4, c = 64, d = 48, elmnt_range=(-2, 2), Nsteps=4, Nsamples=32, torso_iterations=5)
alphaTensor184.to("cuda")
alphaTensor184.load_state_dict(torch.load("models/model_1_test.pt"))

t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)

print(t, r, a)

alphaTensor184.eval()

if True: self_play(alphaTensor184, 4, canonical, 100, num_sim=10, max_actions=10, identifier=1)

if False: 
    for i in tqdm(range(100)):
        alphaTensor184(torch.ones(1, 4, 4, 4).to("cuda"))