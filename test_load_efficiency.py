from torch.distributions import Categorical, Normal
import torch
logits = torch.softmax(torch.randn(7, 8), -1)
print(logits)
dist_dis = Categorical(logits=logits)
print(dist_dis.entropy())