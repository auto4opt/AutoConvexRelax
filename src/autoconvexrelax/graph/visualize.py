import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os

import torch

from autoconvexrelax.paths import OUTPUT_ROOT

TERM_TYPE_SET = [
    'linear', 'quad_diag', 'quad_cross',
    'log', 'log_two_term_sum',
    'abs', 'abs_two_term_sum',
    'sqrt', 'sqrt_two_term_sum',
    'exp', 'exp_two_term_sum',
    'square', 'square_two_term_sum',
    'sin', 'sin_two_term_sum',
    'cos', 'cos_two_term_sum',
    'tan', 'tan_two_term_sum',
    'div', 'div_two_term_sum_num',
    'div_two_term_sum_den', 'div_two_term_sum_both',
    'const', 'other',
    'matrix', 'transpose', 'matmul', 'trace'
]

def visualize_hetero_graph(data, save_path=str(OUTPUT_ROOT / "figures" / "graphs" / "qcqp_graph.png")):
    G = nx.DiGraph()

    # 1. 添加节点
    var_nodes = data["variable"].x.shape[0]
    term_nodes = data["term"].x.shape[0]
    cons_nodes = data["constraint"].x.shape[0]

    for i in range(var_nodes):
        G.add_node(f"v{i}", node_type="variable")

    
    for i in range(term_nodes):
        # ① 取 one‑hot 段（前 21 维）
        one_hot = data["term"].x[i][:len(TERM_TYPE_SET)]
        term_type_id = int(torch.argmax(one_hot).item())
        term_type_str = TERM_TYPE_SET[term_type_id]
        G.add_node(f"t{i}", node_type="term", label=term_type_str)



    constraint_node_ids = set()
    # 先从 term->constraint 边中获取所有目标函数/约束的 ID
    edge_t_c_index = data["term", "in", "constraint"].edge_index
    for _, cid in edge_t_c_index.t().tolist():
        constraint_node_ids.add(cid)

    for cid in constraint_node_ids:
        G.add_node(f"c{cid}", node_type="constraint")



    # 2. 添加边
    def add_edges(edge_index, src_prefix, tgt_prefix, edge_type, color, style="solid", edge_attr=None):
        for idx, (src, tgt) in enumerate(edge_index.t().tolist()):
            edge_label = None
            if edge_attr is not None and edge_type == "uses":
                arr = edge_attr[idx]
                is_num = int(arr[0].item() if hasattr(arr[0], "item") else arr[0])
                is_den = int(arr[1].item() if hasattr(arr[1], "item") else arr[1])
                if is_num == 1 and is_den == 0:
                    edge_label = "numerator"
                elif is_den == 1 and is_num == 0:
                    edge_label = "denominator"
                else:
                    edge_label = None  # 无分式 → 两位都是 0，不标
            G.add_edge(f"{src_prefix}{src}", f"{tgt_prefix}{tgt}",
                    edge_type=edge_type, color=color, style=style, label=edge_label)


    uses_attr = data["variable", "uses", "term"].get("edge_attr")
    add_edges(data["variable", "uses", "term"].edge_index, "v", "t", "uses", "forestgreen", edge_attr=uses_attr)
    add_edges(data["term", "nested", "term"].edge_index, "t", "t", "nested", "gray", style="dashed")
    add_edges(data["term", "in", "constraint"].edge_index, "t", "c", "in", "royalblue")

    # 3. 布局
    # pos = nx.spring_layout(G, seed=42, k=2.5, iterations=100)
    # 3. 手动三分图布局：变量左、项中、约束右
    pos = {}
    x_spacing = 1.5
    y_spacing = 1.0

    # 获取每类节点
    variables = [n for n in G.nodes if n.startswith("v")]
    terms = [n for n in G.nodes if n.startswith("t")]
    constraints = [n for n in G.nodes if n.startswith("c")]

    # 排列每类节点的垂直位置
    for i, node in enumerate(sorted(variables)):
        pos[node] = (-x_spacing, -i * y_spacing)
    # 取 depth:  你 term_feats = type_hot + sign_hot + [depth]
    depth_idx = -1   # 最后一维就是 depth

    for i, node in enumerate(sorted(terms)):
        depth_val = data["term"].x[i][depth_idx].item()
        x_pos = depth_val * 0.4          # 每多一层右移 0.4
        pos[node] = (x_pos, -i * y_spacing)
    for i, node in enumerate(sorted(constraints)):
        pos[node] = (x_spacing, -i * y_spacing)


    # 4. 节点样式
    node_styles = {
        "variable": {"color": "forestgreen", "shape": "o"},
        "term":     {"color": "darkorange", "shape": "d"},
        "constraint": {"color": "dodgerblue", "shape": "s"},
    }

    for ntype, style in node_styles.items():
        nodelist = [n for n in G.nodes if G.nodes[n].get("node_type") == ntype]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodelist,
            node_color=style["color"],
            node_shape=style["shape"],
            node_size=1000,
            alpha=1.0,
            label=ntype
        )

    # 5. 边样式
    for style in {"solid", "dashed"}:
        edgelist = [(u, v) for u, v in G.edges if G[u][v].get("style", "solid") == style]
        nx.draw_networkx_edges(
            G, pos, edgelist=edgelist,
            edge_color=[G[u][v]["color"] for u, v in edgelist],
            style=style,
            arrows=True,
            arrowstyle="->",
            arrowsize=15,
            width=1.5
        )

    # 6. 标签
    node_labels = {}
    for n in G.nodes:
        ntype = G.nodes[n].get("node_type")
        if ntype == "term":
            node_labels[n] = f"{n}\n{G.nodes[n]['label']}"  # 显示 term_type
        else:
            node_labels[n] = n

    nx.draw_networkx_labels(
        G, pos, labels=node_labels,
        font_size=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
    )

    edge_labels = {
        (u, v): G[u][v].get("label", "") for u, v in G.edges if G[u][v].get("label")
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels,
        font_size=7, label_pos=0.5
    )

    # 7. 图例
    legend_elements = [
        Patch(facecolor='forestgreen', label='Variable', edgecolor='k'),
        Patch(facecolor='darkorange', label='Term', edgecolor='k'),
        Patch(facecolor='dodgerblue', label='Constraint', edgecolor='k'),
        Line2D([0], [0], color='forestgreen', lw=2, label='uses'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='nested'),
        Line2D([0], [0], color='royalblue', lw=2, label='in'),
    ]
    plt.legend(handles=legend_elements, loc='lower left', fontsize=8)

    # 8. 保存与展示
    plt.title("Heterogeneous QCQP Graph", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=400)
    print(f"图像已保存为：{os.path.abspath(save_path)}")
    plt.close()

