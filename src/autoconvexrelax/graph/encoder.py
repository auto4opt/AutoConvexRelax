import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv, GATv2Conv, global_mean_pool

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Projector(nn.Module):
    def __init__(self, in_dim, hid=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x): return self.net(x)
    
# ------------------------- Encoder (保持不变) -------------------------
class GNNEncoder(nn.Module):
    def __init__(self, in_dims, hidden_dim=512, out_dim=256, uses_edge_dim=None, p_drop=0.1):
        super().__init__()
        self.emb_var = nn.Linear(in_dims["variable"], hidden_dim)
        self.emb_term = nn.Linear(in_dims["term"], hidden_dim)
        self.emb_constr = nn.Linear(in_dims["constraint"], hidden_dim)

        def block():
            return HeteroConv({
                ('variable','uses','term'): GATv2Conv((-1,-1), hidden_dim, edge_dim=uses_edge_dim, add_self_loops=False),
                ('term','uses_rev','variable'): GATv2Conv((-1,-1), hidden_dim, edge_dim=uses_edge_dim, add_self_loops=False),
                ('term','nested','term'): GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True),
                ('term','nested_rev','term'): GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True),
                ('term','in','constraint'): SAGEConv((-1,-1), hidden_dim, aggr='mean'),
                ('constraint','in_rev','term'): SAGEConv((-1,-1), hidden_dim, aggr='mean'),
            }, aggr='sum')

        self.conv1 = block()
        self.conv2 = block()
        self.norm_var = nn.LayerNorm(hidden_dim)
        self.norm_term = nn.LayerNorm(hidden_dim)
        self.norm_con  = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(p_drop)

        # 用于把 hidden_dim 投到策略所需维度
        self.node_proj = nn.Linear(hidden_dim, out_dim)
        # 全局 token 的投影（把拼接的池化向量投到 out_dim）
        self.global_proj = nn.Linear(hidden_dim * 3, out_dim)

    @staticmethod
    def _merge_missing_node_types(x_prev, x_new):
        """
        HeteroConv may skip node types with no incoming messages for a batch.
        Keep previous features for missing/None node types to avoid None in later layers.
        """
        merged = {}
        for k, v_prev in x_prev.items():
            v_new = x_new.get(k, None) if isinstance(x_new, dict) else None
            merged[k] = v_prev if v_new is None else v_new

        if isinstance(x_new, dict):
            for k, v_new in x_new.items():
                if k not in merged and v_new is not None:
                    merged[k] = v_new
        return merged

    def encode_nodes(self, data):
        """
        返回三类节点的隐藏向量 (未投影)：x_var, x_term, x_con，形状分别为
        [#var, H], [#term, H], [#con, H]
        """
        x = {
            "variable": self.emb_var(data["variable"].x),
            "term":     self.emb_term(data["term"].x),
            "constraint": self.emb_constr(data["constraint"].x),
        }
        edge_attr = {et: data[et].edge_attr for et in data.edge_types if "edge_attr" in data[et]}

        x_after_conv1 = self.conv1(x, data.edge_index_dict, edge_attr_dict=edge_attr)
        x = self._merge_missing_node_types(x, x_after_conv1)
        x = {k: F.relu(v) for k, v in x.items()}

        x_after_conv2 = self.conv2(x, data.edge_index_dict, edge_attr_dict=edge_attr)
        x = self._merge_missing_node_types(x, x_after_conv2)

        x["variable"]   = self.drop(self.norm_var(x["variable"]))
        x["term"]       = self.drop(self.norm_term(x["term"]))
        x["constraint"] = self.drop(self.norm_con(x["constraint"]))
        return x

    def forward(self, data, term_order_index=None, batched: bool = False):
        """
        返回:
          - batched=False: (g: [D],      term_tokens: [n, D])   # 单图
          - batched=True : (g: [B, D],  term_tokens: [N, D])   # 多图(batch)
        """
        x = self.encode_nodes(data)   # dict: "variable"/"term"/"constraint" -> [Ntype, H]

        H = x["term"].size(1)
        device = x["term"].device

        if batched:
            # 计算 batch size（哪个类型都有 batch 字段，用任何一个都行；以 term 为主）
            def get_B(store_name):
                if x[store_name].numel() == 0:
                    return 0
                return int(data[store_name].batch.max().item()) + 1

            B = max(get_B("term"), get_B("variable"), get_B("constraint"))
            if B == 0:
                B = 1  # 兜底

            # 各类节点做每图池化；为空时给零张量
            if x["variable"].numel() > 0:
                v_pool = global_mean_pool(x["variable"], data["variable"].batch)  # [B, H]
            else:
                v_pool = torch.zeros(B, H, device=device)

            if x["term"].numel() > 0:
                t_pool = global_mean_pool(x["term"], data["term"].batch)          # [B, H]
            else:
                t_pool = torch.zeros(B, H, device=device)

            if x["constraint"].numel() > 0:
                c_pool = global_mean_pool(x["constraint"], data["constraint"].batch)  # [B, H]
            else:
                c_pool = torch.zeros(B, H, device=device)

            g_hidden = torch.cat([v_pool, t_pool, c_pool], dim=1)   # [B, 3H]
            g = self.global_proj(g_hidden)                           # [B, D]

            # term tokens（逐节点投影，batch 维通过 data["term"].batch 区分）
            term_hidden = x["term"]                                  # [N, H]
            if term_order_index is not None:
                term_hidden = term_hidden.index_select(0, term_order_index)
            term_tokens = self.node_proj(term_hidden)                # [N, D]
            return g, term_tokens

        else:
            # 单图（旧逻辑）
            v_pool = x["variable"].mean(dim=0, keepdim=True) if x["variable"].numel() > 0 \
                     else torch.zeros(1, H, device=device)
            t_pool = x["term"].mean(dim=0, keepdim=True) if x["term"].numel() > 0 \
                     else torch.zeros(1, H, device=device)
            c_pool = x["constraint"].mean(dim=0, keepdim=True) if x["constraint"].numel() > 0 \
                     else torch.zeros(1, H, device=device)
            g_hidden = torch.cat([v_pool, t_pool, c_pool], dim=1)   # [1, 3H]
            g = self.global_proj(g_hidden)                           # [1, D]

            term_hidden = x["term"]                                  # [n, H]
            if term_order_index is not None:
                term_hidden = term_hidden.index_select(0, term_order_index)
            term_tokens = self.node_proj(term_hidden)                # [n, D]
            return g.squeeze(0), term_tokens
