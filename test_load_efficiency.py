import torch.nn.functional as F
import torch

last_action_dis = torch.randint(0, 8, (12, 20))
agent_to_update = torch.randint(0, 2, (12, 20)).bool().unsqueeze(-1)
print(agent_to_update.expand(-1, -1, 8))
one_hot = F.one_hot(last_action_dis, num_classes=8)
mask = torch.where(agent_to_update.expand(-1, -1, 8), one_hot, torch.zeros_like(one_hot))
print(mask)