# AutoConvexRelax

AutoConvexRelax learns to construct convex relaxations for nonconvex optimization. It converts a symbolic QCQP into a tripartite graph, encodes the evolving formulation, and selects validity-preserving relaxation actions until a convex surrogate is obtained.

This repository contains the implementation used for the manuscript **Learning to Design Convex Relaxations for Nonconvex Optimization**.

## Repository layout

The top level contains only configuration and four functional directories:

```text
code/
├── README.md
├── environment.yml
├── requirements.txt
├── src/autoconvexrelax/   # reusable implementation
├── run/                   # public command-line and SLURM entry points
├── tests/                 # smoke and regression tests
└── outputs/               # figures, logs, checkpoints, and generated data
```

The source package is organized by responsibility:

| Path | Purpose |
| --- | --- |
| `core/` | QCQP representation and convex-relaxation actions |
| `graph/` | Tripartite graph construction, encoding, and visualization |
| `model/` | Learned policy network |
| `problems/` | Training and evaluation instance generators |
| `training/` | Stage-1 shaped-reward and stage-2 solver-guided PPO implementations |
| `evaluation/` | Policy rollout, baselines, solver adapters, and real applications |
| `analysis/` | Result summaries and manuscript figures |
| `tools/` | Environment checks and solver-cache preparation |

## Environment

The reported experiments used Python 3.10.18, PyTorch 2.4.1, CUDA 12.1, and NVIDIA V100 GPUs. Create the environment with either:

```bash
conda env create -f environment.yml
conda activate autoconvexrelax
```

or:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

MOSEK, Gurobi, and SCIP are external dependencies. Their Python bindings and licenses are not distributed with this repository. Keep license files outside the project, for example:

```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
python run/check_environment.py
```

The commands under `run/` configure the `src/` package path automatically.

## Public commands

The repository has six main entry points:

```bash
python run/prepare_data.py --help
python run/train.py --stage 1
python run/train.py --stage 2
python run/infer.py --help
python run/evaluate.py --help
python run/analyze.py --help
python run/plot.py --help
```

Subcommand help is shown when running `prepare_data.py`, `analyze.py`, or `plot.py` without a subcommand.

## Data generation and training

Generate the standard stage-1 and stage-2 datasets:

```bash
python run/prepare_data.py generate all
python run/prepare_data.py generate hybrid
```

The principal workflow expects:

- `vector_all_mix_1600.pkl` for stage 1;
- `vector_finetune_HYBRID_MIX_1200.pkl` for stage 2;
- `root_lb_cache_HYBRID_MIX_1200_root_only.json` for solver-guided reward;
- the split-index JSON saved with each training run;
- the trained checkpoint used for evaluation.

Training settings remain configurable through the existing `APA_*` environment variables. A representative two-stage run is:

```bash
APA_GROUP_UPDATES=200 \
APA_FINETUNE_FILE=outputs/data/vector_all_mix_1600.pkl \
APA_LOG_DIR=outputs/logs/train/stage1 \
python run/train.py --stage 1

APA_GROUP_UPDATES=200 \
APA_FINETUNE_FILE=outputs/data/vector_finetune_HYBRID_MIX_1200.pkl \
APA_ROOT_LB_CACHE_PATH=outputs/data/root_lb_cache_HYBRID_MIX_1200_root_only.json \
APA_LOG_DIR=outputs/logs/train/stage2 \
python run/train.py --stage 2
```

For the reported five-seed cluster workflow:

```bash
sbatch run/slurm/sbatch_train_5seeds_array.sh
```

## Evaluation

A representative held-out comparison is:

```bash
python run/evaluate.py \
  --ckpt /path/to/checkpoint.pt \
  --data outputs/data/vector_finetune_HYBRID_MIX_1200.pkl \
  --split_json /path/to/HYBRID_MIX_1200_split_indices.json \
  --split_use infer \
  --dataset_tag HYBRID_MIX_1200 \
  --n_problems -1 \
  --seed 42 \
  --save_dir outputs/logs/eval_seed_42
```

The comparison algorithms are included in `src/autoconvexrelax/evaluation/baselines.py`:

| Comparison | Implementation |
| --- | --- |
| Gurobi root relaxation | `evaluation/solvers/gurobi.py` and optional cache |
| SCIP root relaxation | `evaluation/solvers/scip.py` and optional cache |
| McCormick endpoint | baseline mode `mccormick` |
| Strengthened SDP endpoint | baseline mode `sdp` plus the relaxation engine |
| Structure-based heuristic | baseline mode `structure` |
| Random policy | baseline mode `random` |

MOSEK solves terminal convex surrogates; Gurobi and SCIP provide solver references.

Fractional instances and real applications use the same public interface:

```bash
python run/prepare_data.py hard-fraction \
  --output outputs/data/fraction_eval.pkl \
  --num-repeat 3 \
  --seed 42

python run/evaluate.py real-applications \
  --ckpt /path/to/checkpoint.pt \
  --device cpu \
  --out_dir outputs/real_applications/paper
```

Cluster evaluation scripts are under `run/slurm/`.

## Analysis and figures

Examples:

```bash
python run/analyze.py summarize --dir outputs/logs/eval_seed_42
python run/analyze.py multiseed --root_dir outputs/logs/multiseed_eval
python run/plot.py training --help
python run/plot.py main
python run/plot.py paper
```

Generated figures are written to `outputs/figures/`; logs, checkpoints, and generated datasets should also remain under `outputs/`.

## Tests

The solver-independent smoke suite covers symbolic actions, baselines, fractional generators, and summaries:

```bash
bash run/smoke_tests.sh
```

Solver-backed evaluation additionally requires the corresponding bindings and licenses.

## Exact numerical reproduction

An archival release should include an immutable artifact bundle containing the exact datasets and splits, selected checkpoints, Gurobi and SCIP caches, raw JSON/CSV outputs, solver versions, and a manifest connecting every table and figure to its seed and command.

## Authors and contact

Jinrun Liu, Bai Yan, Tengfei Liu, Qunfeng Liu, Jian Yang, Shi Cheng, Yuhui Shi, and Qi Zhao.

For questions about the paper or code, contact the corresponding author, Qi Zhao, at [zhaoq@sustech.edu.cn](mailto:zhaoq@sustech.edu.cn). The same address may be used for commercial licensing inquiries.

## License

This software is available under the [PolyForm Noncommercial License 1.0.0](LICENSE.md) (`PolyForm-Noncommercial-1.0.0`). It may be used, modified, and redistributed for noncommercial purposes, including academic research and education. Commercial use requires a separate written license.
