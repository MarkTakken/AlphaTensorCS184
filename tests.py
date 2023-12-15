from architecture import *
from training import *
from mcts import *
from tensorgame import *
from utilities import *
from selfplay import *
from time import time

S = 4
c = 64
d = 48
batches = 32
elmnt_range = (-2, 2)
Nsamples = 10
state = torch.zeros(batches, S, S, S)
grid = torch.zeros(batches, S, S, c)
rod = torch.zeros(S, 2*S, c)
action = torch.zeros(batches, S, dtype=int)
e = torch.zeros(batches, 3*S*S, c)
tokens = torch.zeros(batches, S-1, c)

if __name__ == '__main__':
    if False:
        attention = Attention(c, c, device=torch.device('cpu'))
        print("Attention test 1:", attention(rod, rod).shape)
        attention = Attention(c, c, device=torch.device('cpu'))
        print("Attention test 2:", attention(tokens, e).shape)
        attention = Attention(c, c, device=torch.device('cpu'), causal_mask=True)
        print("Attention test 3:", attention(tokens, tokens).shape)

    if False:
        attentive_modes = AttentiveModes(S, c, device=torch.device('cpu'))
        result = attentive_modes(grid, grid, grid)
        print("Attentive modes test 1:", len(result), result[0].shape)

    if False:
        torso = Torso(S, c, 2, device=torch.device('cpu'))
        print("Torso test 1:", torso(state).shape)

    if False:
        value_head = ValueHead(c, d)
        print("Value test 1:", value_head(e).shape)

    if False:
        policy_head = PolicyHead(S, elmnt_range, S, c, device=torch.device('cpu'))
        print("Policy test 1:", policy_head.predict_logits(action, e).shape)
        policy_head.train()
        print("Policy test 2:", policy_head(e, g=action).shape)
        policy_head.eval()
        actions, probs = policy_head(e[0:1], Nsamples=Nsamples)
        print("Policy test 3:", actions.shape, probs.tolist())

    if False:
        alphatensor = AlphaTensor184(S, c, d, elmnt_range, S, Nsamples, device=torch.device('cpu'))
        alphatensor.train()
        value, logits = alphatensor(state, g=action)
        print("AlphaTensor test 1:", value.shape, logits.shape)
        alphatensor.eval()
        value, policy = alphatensor(state[0:1])
        print("AlphaTensor test 2:", value.shape, policy.actions.shape, policy.probs.shape)

    if False:
        alphatensor.train()
        pred_value, pred_logits = alphatensor(state, g=action)
        true_value = torch.zeros(batches)
        print("Loss test 1:", loss_fn(pred_logits, action, pred_value, true_value, device='cpu'))

    if False:
        alphatensor.eval()
        state2 = TensorGame(torch.ones(S, S, S), 5)
        mcts = MCTS(state2, alphatensor)
        mcts.search(10, 100)
        print("Search test 1: Done")
        state3 = TensorGame(torch.ones(S, S, S), 50)
        mcts = MCTS(state3, alphatensor)
        mcts.search(10, 100)
        print("Search test 2: Done")
        print("Search test 3:", mcts.get_action_probs(0.9))
        mcts.search_and_play(10, 1, 0.9)
        print("Search test 4: Done")

    if False:
        M = change_of_basis(4, torch.tensor([-2,-1,0,1,2]), torch.tensor([0.1,0.1,0.6,0.1,0.1]))
        print("COB Test 1:\n", M)
        newstate = apply_COB(torch.ones(4,4,4,dtype=int),M)
        print("COB Test 2:\n", newstate)
        print("COB Test 3:\n", invert_COB(newstate, M))

    if False:
        alphatensor = AlphaTensor184(S, c, d, elmnt_range, S, Nsamples, torso_iterations=5, device='cuda')
        alphatensor.to('cuda')
        alphatensor.eval()
        print(self_play(alphatensor, S, 20, device='cpu', num_sim=16, max_actions=10))

    if False:
        alphatensor = AlphaTensor184(S, c, d, elmnt_range, S, Nsamples, torso_iterations=5, device='cpu')
        alphatensor.train()
        nn_state = torch.zeros((1024,4,4,4))
        g = torch.zeros(1024,4,dtype=int)
        T = time()
        alphatensor(nn_state, g=g)
        print(time()-T)
        alphatensor = AlphaTensor184(S, c, d, elmnt_range, S, Nsamples, torso_iterations=5, device='mps')
        alphatensor.to('mps')
        alphatensor.train()
        nn_state = torch.zeros((1024,4,4,4), device='mps')
        g = torch.zeros(1024,4,dtype=int,device='mps')
        T = time()
        alphatensor(nn_state, g=g)
        print(time()-T)

    if True:
        alphatensor = AlphaTensor184(S, c, d, elmnt_range, S, Nsamples, torso_iterations=5, device='cpu')
        alphatensor.eval()
        print(self_play_parallel(4, alphatensor, S, 20, num_sim=16, max_actions=10, device='cpu'))