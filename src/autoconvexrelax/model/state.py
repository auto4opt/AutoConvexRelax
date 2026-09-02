from __future__ import annotations

import torch
import torch.nn as nn
import sympy as sp
from torch_geometric.data import HeteroData

from autoconvexrelax.graph.encoder import GNNEncoder
from autoconvexrelax.graph.conversion import qcqp_to_heterodata, decompose_num_den_vars  
try:
    from sympy.matrices.expressions.matexpr import MatrixElement
except Exception:
    MatrixElement = None

class StateRepresentation(nn.Module):
    """
    -- 新版说明 --
    输入:
      problem: QCQPProblem 对象
    输出:
      seq: [1, 1+k, d_model] = [GLOBAL] + [non_convex_term_1, ..., non_convex_term_k]
           (k = 问题中的非凸项数量)
      non_convex_indices: [k] (torch.long)
           非凸项在原始外层项列表中的索引 (0-based)。用于将模型的动作映射回环境 term_id。
    """
    def __init__(self, d_embed: int, d_model: int, device: str):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.device  = device

        self.gnn: GNNEncoder | None = None
        self.term_proj   = None
        self.global_proj = None
        self._nan_reported_once = False

    def _ensure_modules(self, data: HeteroData, gnn_out_dim: int = 256, gnn_hidden: int = 512):
        if self.gnn is not None:
            return
        in_dims = {
            "variable":   int(data["variable"].x.size(1)),
            "term":       int(data["term"].x.size(1)),
            "constraint": int(data["constraint"].x.size(1)),
        }
        uses_edge_dim = None
        et = ('variable', 'uses', 'term')
        if 'edge_attr' in data[et]:
            uses_edge_dim = int(data[et].edge_attr.size(1))

        self.gnn = GNNEncoder(
            in_dims, hidden_dim=gnn_hidden, out_dim=gnn_out_dim,
            uses_edge_dim=uses_edge_dim, p_drop=0.1
        ).to(self.device)

        self.term_proj   = nn.Linear(gnn_out_dim, self.d_model) if gnn_out_dim != self.d_model else nn.Identity()
        self.global_proj = nn.Linear(gnn_out_dim, self.d_model) if gnn_out_dim != self.d_model else nn.Identity()

    @staticmethod
    def _get_flags(problem):
        flags = problem.get_convexity_flags()
        if not isinstance(flags, torch.Tensor):
            flags = torch.tensor(flags, dtype=torch.bool)
        return flags

    @staticmethod
    def _base_matrix_symbols(expr):
        mats = set(expr.atoms(sp.MatrixSymbol))
        for tr in expr.atoms(sp.Transpose):
            if isinstance(tr.arg, sp.MatrixSymbol):
                mats.add(tr.arg)
        if MatrixElement is not None:
            for me in expr.atoms(MatrixElement):
                parent = getattr(me, "parent", None) or getattr(me, "base", None) or me.args[0]
                if isinstance(parent, sp.MatrixSymbol):
                    mats.add(parent)
        return mats

    @classmethod
    def _term_vtype_flags(cls, problem, expr):
        names = {str(s.name) for s in expr.free_symbols}
        for ms in cls._base_matrix_symbols(expr):
            names.add(str(ms.name))

        has_cont = has_int = has_bin = 0.0
        for vname in names:
            v_info = problem.variables.get(vname) or problem.matrix_variables.get(vname)
            if v_info is None:
                continue
            vtype = getattr(v_info, "vtype", "continuous")
            if vtype == "continuous":
                has_cont = 1.0
            elif vtype == "integer":
                has_int = 1.0
            elif vtype == "binary":
                has_bin = 1.0
        return [has_cont, has_int, has_bin]

    @staticmethod
    def _sanitize_tensor(t: torch.Tensor, clip_val: float = 1e6) -> torch.Tensor:
        t = torch.nan_to_num(t, nan=0.0, posinf=clip_val, neginf=-clip_val)
        return torch.clamp(t, min=-clip_val, max=clip_val)

    def _sanitize_graph_data(self, data: HeteroData) -> HeteroData:
        for ntype in data.node_types:
            if "x" in data[ntype]:
                data[ntype].x = self._sanitize_tensor(data[ntype].x)
        for et in data.edge_types:
            if "edge_attr" in data[et]:
                data[et].edge_attr = self._sanitize_tensor(data[et].edge_attr)
        return data

    def _report_non_finite_once(self, problem, data: HeteroData, stage: str, t: torch.Tensor):
        if self._nan_reported_once:
            return
        self._nan_reported_once = True

        pname = getattr(problem, "name", "<unknown_problem>")
        n_nonfinite = int((~torch.isfinite(t)).sum().item())
        print(f"[StateRepresentation][NON_FINITE] stage={stage} problem={pname} non_finite={n_nonfinite}")

        node_stats = {}
        for ntype in data.node_types:
            n = int(data[ntype].x.size(0)) if "x" in data[ntype] else 0
            node_stats[ntype] = n
        edge_stats = {}
        for et in data.edge_types:
            edge_stats[str(et)] = int(data[et].edge_index.size(1))
        print(f"[StateRepresentation][NON_FINITE] node_stats={node_stats}")
        print(f"[StateRepresentation][NON_FINITE] edge_stats={edge_stats}")

    def forward(self, x):
        """
        x: 直接传入 QCQPProblem 对象
        """
        problem = x
        data: HeteroData = qcqp_to_heterodata(problem).to(self.device)
        data = self._sanitize_graph_data(data)

        # 1) 初始化 GNN (不变)
        self._ensure_modules(data)

        # 2) 计算所有 term 节点 embedding (不变)
        g_token, term_tokens_all = self.gnn(data)
        if not torch.isfinite(g_token).all():
            self._report_non_finite_once(problem, data, "gnn_g_token", g_token)
            g_token = self._sanitize_tensor(g_token)
        if not torch.isfinite(term_tokens_all).all():
            self._report_non_finite_once(problem, data, "gnn_term_tokens_all", term_tokens_all)
            term_tokens_all = self._sanitize_tensor(term_tokens_all)

        g_token = self.global_proj(g_token)
        term_tokens_all = self.term_proj(term_tokens_all)
        if not torch.isfinite(g_token).all():
            self._report_non_finite_once(problem, data, "global_proj_g_token", g_token)
            g_token = self._sanitize_tensor(g_token)
        if not torch.isfinite(term_tokens_all).all():
            self._report_non_finite_once(problem, data, "term_proj_term_tokens_all", term_tokens_all)
            term_tokens_all = self._sanitize_tensor(term_tokens_all)

        # 3) 取出所有外层项的 embedding (不变)
        outer_index = data["term"].outer_index.to(self.device)
        term_tokens = term_tokens_all.index_select(0, outer_index)
        
        # 3.5) 直接从外层项表达式重算变量类型特征，避免图构建阶段特征与引擎判定不一致
        sorted_term_ids = sorted(problem.id_to_item.keys())
        outer_vtypes_list = []
        frac_flags = []
        for term_id in sorted_term_ids:
            expr, _location = problem.id_to_item[term_id]
            outer_vtypes_list.append(self._term_vtype_flags(problem, expr))
            _num_vars, den_vars, _has_frac_raw = decompose_num_den_vars(expr)
            has_symbolic_den = (len(den_vars) > 0)
            frac_flags.append(1.0 if has_symbolic_den else 0.0)

        outer_vtypes = torch.tensor(
            outer_vtypes_list, dtype=term_tokens.dtype, device=self.device
        ) if outer_vtypes_list else torch.zeros((0, 3), dtype=term_tokens.dtype, device=self.device)

        frac_flags = torch.tensor(
            frac_flags, dtype=outer_vtypes.dtype, device=self.device
        )

        # 4) 获取凸性标志，并找出非凸项的索引
        flags = self._get_flags(problem).to(self.device)
        if term_tokens.size(0) != flags.shape[0] or outer_vtypes.size(0) != flags.shape[0]:
            raise RuntimeError(
                f"Outer-term alignment mismatch: term_tokens={term_tokens.size(0)}, "
                f"outer_vtypes={outer_vtypes.size(0)}, flags={flags.shape[0]}"
            )
        non_convex_mask = ~flags  # True 代表非凸

        num_outer_terms = flags.shape[0]
        original_indices = torch.arange(num_outer_terms, device=self.device)
        non_convex_indices = original_indices[non_convex_mask]

        # 5) 根据 non_convex_indices 筛选出非凸项的 embedding
        non_convex_tokens = term_tokens.index_select(0, non_convex_indices)
        
        # 5.5) 非凸项的显式变量类型 + is_frac 一起拿出来
        non_convex_vtypes = outer_vtypes.index_select(0, non_convex_indices)          # [k, 3]
        non_convex_frac   = frac_flags.index_select(0, non_convex_indices).unsqueeze(-1)  # [k, 1]
        non_convex_vtypes = torch.cat([non_convex_vtypes, non_convex_frac], dim=-1)   # [k, 4]
        if not torch.isfinite(non_convex_vtypes).all():
            self._report_non_finite_once(problem, data, "non_convex_vtypes", non_convex_vtypes)
            non_convex_vtypes = self._sanitize_tensor(non_convex_vtypes)

        # 6) 构建新的、只包含非凸项的序列
        # 如果没有非凸项，序列只包含全局 token
        seq = torch.cat([g_token.unsqueeze(0), non_convex_tokens], dim=0).unsqueeze(0)
        if not torch.isfinite(seq).all():
            self._report_non_finite_once(problem, data, "seq", seq)
            seq = self._sanitize_tensor(seq)
        
        # --- DEBUG ---
        # print(f"[StateRepresentation] seq.shape = {seq.shape}, non_convex_indices = {non_convex_indices.tolist()}")
        
        
        # # --- DEBUG 代码开始: 计算并打印非凸项之间的余弦相似度 ---
        # num_non_convex = non_convex_tokens.shape[0]
        # if num_non_convex > 1:
        #     print("\n" + "="*40 + " 余弦相似度诊断 " + "="*40)
        #     print(f"问题 '{problem.name}' 中发现 {num_non_convex} 个非凸项，计算它们之间的两两相似度:")

        #     # 步骤1: 获取非凸项对应的 SymPy 表达式字符串，方便阅读
        #     non_convex_term_exprs = []
        #     # problem.id_to_item 的 key 是从 1 开始的
        #     # non_convex_indices 是 0-based index
        #     for idx in non_convex_indices:
        #         term_id = idx.item() + 1
        #         expr, location = problem.id_to_item[term_id]
        #         non_convex_term_exprs.append(f"'{expr}' at {location}")

        #     # 步骤2: 高效计算两两余弦相似度矩阵
        #     # [k, d] -> [k, 1, d] vs [1, k, d] => [k, k]
        #     sim_matrix = torch.nn.functional.cosine_similarity(
        #         non_convex_tokens.unsqueeze(1), 
        #         non_convex_tokens.unsqueeze(0), 
        #         dim=2
        #     )

        #     # 步骤3: 打印出上三角矩阵的结果 (避免重复)
        #     for i in range(num_non_convex):
        #         for j in range(i + 1, num_non_convex):
        #             similarity = sim_matrix[i, j].item()
        #             print(f"  - Sim({non_convex_term_exprs[i]}, \n         {non_convex_term_exprs[j]}) = {similarity:.4f}")
            
        #     print("="*100 + "\n")
        # # --- DEBUG 代码结束 ---
        # print(f"[StateRepresentation] Generated state with seq.shape = {seq.shape}, non_convex_indices = {non_convex_indices.tolist()}, non_convex_vtypes = {non_convex_vtypes}")
        return {
            "seq": seq,
            "non_convex_indices": non_convex_indices,
            "non_convex_vtypes": non_convex_vtypes,
        }
        
