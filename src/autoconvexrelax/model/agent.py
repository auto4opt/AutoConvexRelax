import torch
from torch import nn

from autoconvexrelax.model.agent_layer import AgentLayer
# 继续用同一路径下你刚改好的 StateRepresentation
from autoconvexrelax.model.state import StateRepresentation

class Agent(nn.Module):
    def __init__(self, d_model, d_embed, d_actions, n_head, ffn_hidden, drop_prob, device):
        super().__init__()
        self.device = device
        # 现在的 StateRepresentation(problem) → (seq[1,1+n,D], n, flags[n])
        self.state_representation = StateRepresentation(
            d_model=d_model,
            d_embed=d_embed,
            device=device,
        )
        self.agent = AgentLayer(
            d_model=d_model,
            d_actions=d_actions,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            drop_prob=drop_prob,
        )
        
    def forward(self, x, mode="train", problem_type=None, non_convex_vtypes=None):
        # 注意：m 恒等于 0（因为序列是 [GLOBAL] + terms，没有 constraints 段）
        return self.agent(
             state_repr=x,
             mode=mode,
             problem_type=problem_type,
             non_convex_vtypes=non_convex_vtypes # <-- 将接收到的新信息传递下去
         )