## Main is meant to control the overall running of the program, switching between self-play and training

import torch
from architecture import *
from mcts import *
from training import *

def main(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Either initialize or load the model
    # Load the buffer

    # Begin the self-play/training Loop
    ## Sample from the buffer // Transformations
    ## Train on the buffer given hyperparams for main
    ## Save the model to the next iteration
    ## Self play based on the hyperparams
    ## Save played trajectories with changes to the buffer
    ## Save successful trajectories to a file
    ## Print out the number of successful trajectories and total loss of trajectories sampled 
    ## Repeat
    # End the loop after x iterations

    # Report on successful trajectories found during time of running

if __name__ == "__main__":
    main()
