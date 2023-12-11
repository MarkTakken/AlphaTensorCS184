from architecture import *
from training import *
from mcts import *

S = 3
c = 64
d = 10
batches = 32
elmnt_range = (-2, 2)
Nsamples = 5
state = torch.zeros(batches, S, S, S)
grid = torch.zeros(batches, S, S, c)
rod = torch.zeros(S, 2*S, c)
action = torch.zeros(batches, S, dtype=int)
e = torch.zeros(batches, 3*S*S, c)
tokens = torch.zeros(batches, S-1, c)

attention = Attention(c, c)
print("Attention test 1:", attention(rod, rod).shape)
attention = Attention(c, c)
print("Attention test 2:", attention(tokens, e).shape)
attention = Attention(c, c, causal_mask=True)
print("Attention test 3:", attention(tokens, tokens).shape)

attentive_modes = AttentiveModes(S, c)
result = attentive_modes(grid, grid, grid)
print("Attentive modes test 1:", len(result), result[0].shape)

torso = Torso(S, c, 2)
print("Torso test 1:", torso(state).shape)

value_head = ValueHead(c, d)
print("Value test 1:", value_head(e).shape)

policy_head = PolicyHead(S, elmnt_range, S, c)
print("Policy test 1:", policy_head.predict_logits(action, e).shape)
policy_head.train()
print("Policy test 2:", policy_head(e, g=action).shape)
policy_head.eval()
actions, probs = policy_head(e[0:1], Nsamples=Nsamples)
print("Policy test 3:", actions.shape, probs.tolist())

alphatensor = AlphaTensor184(S, c, d, elmnt_range, S, Nsamples)
alphatensor.train()
value, logits = alphatensor(state, g=action)
print("AlphaTensor test 1:", value.shape, logits.shape)
alphatensor.eval()
value, policy = alphatensor(state[0:1])
print("AlphaTensor test 2:", value.shape, policy.actions.shape, policy.probs.shape)

alphatensor.train()
pred_value, pred_logits = alphatensor(state, g=action)
true_value = torch.zeros(batches)
print("Loss test 1:", loss(pred_logits, action, pred_value, true_value))

alphatensor.eval()
state2 = torch.ones(1, S, S, S)
mcts = MCTS(state2, alphatensor)
mcts.search(10, 1)