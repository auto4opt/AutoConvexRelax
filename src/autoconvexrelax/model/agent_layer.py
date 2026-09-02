import torch
from torch import nn

from autoconvexrelax.model.ffn import FFN
from autoconvexrelax.model.layer_norm import LayerNorm
from autoconvexrelax.model.multi_head_attention import MultiHeadAttention

class AgentLayer(nn.Module):
    def __init__(self, d_model, d_actions, ffn_hidden, n_head, drop_prob):
        super(AgentLayer, self).__init__()
        
        self.last_action_logits = None
        self.d_model = d_model
        self.d_actions = d_actions
        
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        
        self.ffn1 = FFN(d_model=d_model, d_out=1, hidden=ffn_hidden, drop_prob=drop_prob)
        self.softmax1 = nn.Softmax(dim=-1)
        
        ffn2_in_dim = d_model + 4
        self.ffn2 = FFN(d_model=ffn2_in_dim, d_out=d_actions, hidden=ffn_hidden, drop_prob=drop_prob)
        self.softmax2 = nn.Softmax(dim=-1)
        
        self.scalar_only_actions = {7, 8}
        self.vector_only_actions = {7, 8}
        
        self.FILL = -1e9

    @staticmethod
    def _sanitize_tensor(x: torch.Tensor, clip_val: float = 1e6) -> torch.Tensor:
        """
        Replace non-finite values and clamp magnitude to keep logits stable.
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=clip_val, neginf=-clip_val)
        return torch.clamp(x, min=-clip_val, max=clip_val)

    def add_problem_type_mask(self, action_logits, problem_type):
        if problem_type == 'scalar':
            mask_indices = list(self.vector_only_actions)
        elif problem_type == 'vector':
            mask_indices = list(self.scalar_only_actions)
        else:
            return action_logits
        
        if action_logits.dim() == 1:
            action_logits = action_logits.unsqueeze(0)
        
        for idx in mask_indices:
             if 0 <= idx < action_logits.shape[1]:
                #  action_logits[:, idx] = float('-inf')
                action_logits[:, idx] = self.FILL

        return action_logits.squeeze(0) if action_logits.shape[0] == 1 and action_logits.dim() > 1 else action_logits

    def forward(self, state_repr, mode="train", problem_type=None, non_convex_vtypes=None):      
        if state_repr.dim() == 2:
            state_repr = state_repr.unsqueeze(0)
        state_repr = self._sanitize_tensor(state_repr)

        # === 阶段一: 选择 Term ===
        attn_output = self.self_attention(state_repr, state_repr, state_repr)
        attn_output = self.norm1(attn_output)
        attn_output = self._sanitize_tensor(attn_output)
        
        batch_size = attn_output.size(0)
        action_logits = self.ffn1(attn_output).squeeze(-1)
        action_logits = self._sanitize_tensor(action_logits)

        n = state_repr.size(1) - 1
        # if n > 0:
        #     valid_mask = torch.zeros_like(action_logits, dtype=torch.bool, device=action_logits.device)
        #     valid_mask[:, 1:] = True
        #     action_logits = action_logits.masked_fill(~valid_mask, self.FILL)
        # else:
        #     action_logits.fill_(self.FILL)

        valid_mask = torch.zeros_like(action_logits, dtype=torch.bool, device=action_logits.device)
        if n <= 0:
            # seq 只有 1 个 token：只能选 location=0（全局 token）
            valid_mask[:, 0] = True
        else:
            # seq 有非凸 token：禁止 location=0，只能在非凸 token 上做动作
            valid_mask[:, 1:] = True
            
        action_logits = action_logits.masked_fill(~valid_mask, self.FILL)
        action_logits = self._sanitize_tensor(action_logits)
        
        self.last_action_logits = action_logits.detach().clone()
        
        action_location_probs = self.softmax1(action_logits)
        action_dist = torch.distributions.Categorical(logits=action_logits)

        if mode == "train":
            action_location = action_dist.sample()
        else:
            action_location = torch.argmax(action_location_probs, dim=-1)
        
        log_prob_action_location = action_dist.log_prob(action_location)

        # === 阶段二: 为选中的 Term 决策 Action ID ===
        batch_idx = torch.arange(batch_size, device=action_location.device)
        selected_term_vector = attn_output[batch_idx, action_location, :]
        
        # action_location==0 表示整题 token，没有对应 non_convex_vtypes，用全 0
        if int(action_location.item()) == 0:
            selected_vtype_features = torch.zeros((1, 4), device=attn_output.device, dtype=attn_output.dtype)
        else:
            vtype_idx_in_list = int((action_location - 1).item())
            selected_vtype_features = non_convex_vtypes[vtype_idx_in_list].unsqueeze(0)  # [1,4]


        final_decision_input = torch.cat([selected_term_vector, selected_vtype_features], dim=-1)
        final_decision_input = self._sanitize_tensor(final_decision_input)
        
        action_id_logits = self.ffn2(final_decision_input)
        action_id_logits = self._sanitize_tensor(action_id_logits)

        if problem_type is not None:
            action_id_logits = self.add_problem_type_mask(action_id_logits, problem_type)
            
        
        # ======= 新增：统一为 [B, A]，避免 size(1) 越界 =======
        if action_id_logits.dim() == 1:
            action_id_logits = action_id_logits.unsqueeze(0)
        
        # === 按 action_location + vtype_features 决定动作可用集合 ===
        # 约定（与你现在 ffn2_in_dim=d_model+4 一致）：
        # selected_vtype_features = [has_cont, is_integer, is_binary, has_fraction]
        # 你若编码不同，只需改下面 has_discrete/has_fraction 的取法。
        loc0 = (int(action_location.item()) == 0)

        if loc0:
            # problem token：只允许 5/6
            allowed = {5, 6}
        else:
            # 非凸项 token：根据特征选择 allowed
            has_cont = bool(selected_vtype_features[0, 0].item() > 0.5)
            has_integer = bool(selected_vtype_features[0, 1].item() > 0.5)
            has_binary  = bool(selected_vtype_features[0, 2].item() > 0.5)
            has_fraction = bool(selected_vtype_features[0, 3].item() > 0.5)

            has_discrete = has_integer or has_binary

            if has_discrete:
                allowed = {0}          # relax_integrality
            elif has_fraction:
                allowed = {1}          # remove_fraction
            else:
                allowed = {2, 3, 4}    # mccormick / sdp / alphaBB

        # 统一按 allowed 做 mask
        mask = torch.zeros_like(action_id_logits, dtype=torch.bool)
        for a in allowed:
            if 0 <= a < action_id_logits.size(1):
                mask[:, a] = True
        action_id_logits = action_id_logits.masked_fill(~mask, self.FILL)
        action_id_logits = self._sanitize_tensor(action_id_logits)


        action_id_probs = self.softmax2(action_id_logits)
        action_id_dist = torch.distributions.Categorical(logits=action_id_logits)

        if mode == "train":
            action_id = action_id_dist.sample()
        else:
            action_id = torch.argmax(action_id_probs, dim=-1)
            
        log_prob_action_id = action_id_dist.log_prob(action_id)
        
        self.saved_log_probs = log_prob_action_id + log_prob_action_location
        
        logp_loc_all = action_logits
        logp_id_all = action_id_logits
        
        return (action_id, action_id_probs, action_location, action_location_probs, 
                logp_id_all, logp_loc_all, log_prob_action_id, log_prob_action_location)
