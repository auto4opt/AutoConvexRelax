import random
import hashlib
from dataclasses import dataclass

import sympy as sp
from autoconvexrelax.core.relaxation import RelaxationEngine


@dataclass(frozen=True)
class QuadraticCouplingStats:
    num_active_variables: int
    offdiag_nnz: int
    offdiag_density: float


def _combine_all_exprs(prob):
    """Combine objective + all constraint LHS/RHS into one SymPy expression."""
    exprs = []
    if prob.obj_expr is not None:
        exprs.append(prob.obj_expr)
    for c in getattr(prob, "constraints", []):
        exprs.append(c.expr)
        if isinstance(c.rhs, sp.Expr):
            exprs.append(c.rhs)
    if not exprs:
        return sp.Integer(0)
    return sp.Add(*exprs)


def _collect_all_exprs(prob):
    """Collect objective + all constraint LHS/RHS as separate SymPy expressions."""
    exprs = []
    if prob.obj_expr is not None:
        exprs.append(prob.obj_expr)
    for c in getattr(prob, "constraints", []):
        exprs.append(c.expr)
        if isinstance(c.rhs, sp.Expr):
            exprs.append(c.rhs)
    return exprs


def _term_has_fraction(term) -> bool:
    """Detect variable-dependent denominator: any Pow(base, exp<0) with base containing symbols."""
    try:
        for pw in term.atoms(sp.Pow):
            exp = pw.exp
            if exp.is_number and exp.is_negative:
                if pw.base.free_symbols:
                    return True
    except Exception:
        return False
    return False


def _collect_fraction_terms(prob):
    prob.map_all_terms()
    frac_terms = []
    for _, (term, loc) in sorted(prob.id_to_item.items()):
        if _term_has_fraction(term):
            frac_terms.append((loc, term))
    return frac_terms


def _collect_nonconvex_terms(prob):
    prob.map_all_terms()
    classes = prob.get_convexity_classes()
    items = list(sorted(prob.id_to_item.items()))
    nonconvex = []
    for (id_, (term, loc)), cls in zip(items, classes):
        if cls == 3:
            nonconvex.append((loc, term))
    return nonconvex


def _action_changed(last_rewrite) -> bool:
    if not isinstance(last_rewrite, dict):
        return False
    return sp.srepr(last_rewrite.get("old")) != sp.srepr(last_rewrite.get("new"))


def _finite_symbols(term):
    try:
        return sorted(list(term.free_symbols), key=lambda s: str(s))
    except Exception:
        return []


def quadratic_coupling_stats(term) -> QuadraticCouplingStats:
    """
    Compute the H1 density statistic for a scalar quadratic expression.

    For x^T Q x, offdiag_nnz counts nonzero entries in offdiag(Q). A cross
    monomial x_i*x_j therefore contributes two entries, Q_ij and Q_ji.
    """
    symbols = _finite_symbols(term)
    if not symbols:
        return QuadraticCouplingStats(0, 0, 0.0)

    try:
        expr = sp.expand(term)
        poly = sp.Poly(expr, *symbols)
    except Exception:
        return QuadraticCouplingStats(0, 0, 0.0)

    active = set()
    offdiag_nnz = 0
    for monom, coeff in poly.terms():
        if coeff == 0 or sum(monom) != 2:
            continue
        nz = [idx for idx, deg in enumerate(monom) if deg > 0]
        for idx in nz:
            active.add(symbols[idx])
        if len(nz) == 2 and monom[nz[0]] == 1 and monom[nz[1]] == 1:
            # Symmetric Q has both Q_ij and Q_ji nonzero for one cross term.
            offdiag_nnz += 2

    m = len(active)
    denom = max(1, m * (m - 1))
    return QuadraticCouplingStats(
        num_active_variables=m,
        offdiag_nnz=offdiag_nnz,
        offdiag_density=float(offdiag_nnz) / float(denom),
    )


def choose_structure_action(term, k_min: int = 3, tau_density: float = 0.5) -> str:
    stats = quadratic_coupling_stats(term)
    if stats.num_active_variables >= k_min and stats.offdiag_density >= tau_density:
        return "sdp_relaxation"
    return "mccormick_relaxation"


def derive_random_baseline_seed(base_seed: int, rollout: int, problem_name: str) -> int:
    payload = f"{int(base_seed)}:{int(rollout)}:{problem_name}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _needs_structural_relaxation(prob, loc, cls) -> bool:
    """
    Detect terms that are convex as expressions but still appear in a
    structurally nonconvex position for the solver interface.

    Examples:
    - maximize convex quadratic objective
    - convex nonlinear term on the LHS of a >= constraint
    - convex nonlinear term inside an equality constraint
    """
    if cls not in (0, 1):
        return False

    if loc == "Objective":
        return getattr(prob, "obj_sense", "min") != "min"

    if not isinstance(loc, str) or not loc.startswith("Constraint_") or not loc.endswith("_LHS"):
        return False

    try:
        idx = int(loc.split("_")[1]) - 1
        cons = prob.constraints[idx]
    except Exception:
        return False

    return cons.sense in (">=", ">", "=", "==")


def _collect_relaxation_targets_with_ids(prob):
    """
    Collect terms that should be relaxed by the heuristic baseline.

    Besides explicitly nonconvex terms (class 3), we also include
    "structurally nonconvex" convex terms such as reverse-convex constraints.
    """
    prob.map_all_terms()
    classes = prob.get_convexity_classes()
    items = list(sorted(prob.id_to_item.items()))
    targets = []
    for (id_, (term, loc)), cls in zip(items, classes):
        if cls == 3 or _needs_structural_relaxation(prob, loc, cls):
            targets.append((id_, loc, term))
    return sorted(targets, key=_target_order_key)


def _target_order_key(target):
    """H1 order: objective first, then constraints, then original term_id."""
    id_, loc, _term = target
    if loc == "Objective":
        return (0, -1, id_)
    if isinstance(loc, str) and loc.startswith("Constraint_"):
        try:
            idx = int(loc.split("_")[1])
        except Exception:
            idx = 10**9
        return (1, idx, id_)
    return (2, 10**9, id_)


def _collect_relaxation_targets(prob):
    return [(loc, term) for _, loc, term in _collect_relaxation_targets_with_ids(prob)]


def _configure_baseline_engine() -> RelaxationEngine:
    engine = RelaxationEngine()
    # Keep baseline strict: disable engine-side strengthening knobs.
    engine.enable_bt_warmup = False
    engine.enable_bt_before_global_cut = False
    engine.enable_mccormick_square_midpoint_tangent = False
    engine.enable_mccormick_square_nonneg_cut = False
    engine.enable_mccormick_link_w_to_sdp = False
    return engine


def _apply_mandatory_preprocessing(prob, engine, enable_relax_integrality: bool, enable_remove_fraction: bool):
    # NOTE: do not sum all expressions; they may cancel to a constant (losing variables).
    if enable_relax_integrality:
        for ex in _collect_all_exprs(prob):
            try:
                engine.apply_action(prob, location="GLOBAL", sub_expr=ex, action_type="relax_integrality")
            except Exception:
                continue

    if enable_remove_fraction:
        for loc, term in _collect_fraction_terms(prob):
            try:
                engine.apply_action(prob, location=loc, sub_expr=term, action_type="remove_fraction")
            except Exception:
                continue


def _try_action(engine, prob, loc, term, action_type: str) -> bool:
    try:
        last = engine.apply_action(prob, location=loc, sub_expr=term, action_type=action_type)
    except Exception:
        return False
    return _action_changed(last)


def _apply_action_strict(engine, prob, loc, term, action_type: str) -> bool:
    last = engine.apply_action(prob, location=loc, sub_expr=term, action_type=action_type)
    return _action_changed(last)


def apply_heuristic_relaxation(
    prob,
    mode: str = "mccormick",
    max_passes: int = 20,
    enable_relax_integrality: bool = True,
    enable_remove_fraction: bool = True,
    k_min: int = 3,
    tau_density: float = 0.5,
    random_seed: int = 0,
    random_include_qcr: bool = True,
):
    """
    Heuristic relaxation (baseline defaults):
      1) relax integrality (on by default)
      2) remove fractions (on by default)
      3) relax nonconvex terms according to the selected baseline mode

    Structure mode is strict: it applies the action selected by H1 and does not
    fall back to another relaxation or swallow action failures.
    Random mode is also strict: each step samples one target and one action
    from the action library, applies it once, and does not fall back to another
    action if the sampled action is ineffective.
    """
    engine = _configure_baseline_engine()
    _apply_mandatory_preprocessing(prob, engine, enable_relax_integrality, enable_remove_fraction)

    # 3) relax nonconvex terms
    rng = random.Random(random_seed)
    for _ in range(max_passes):
        target_terms = _collect_relaxation_targets_with_ids(prob)
        if not target_terms:
            break

        if mode == "random":
            candidates = list(target_terms)
            rng.shuffle(candidates)
            base_actions = ["mccormick_relaxation", "sdp_relaxation"]
            if random_include_qcr:
                base_actions.append("qcr")
            actions = list(base_actions)
            rng.shuffle(actions)

            _, loc, term = candidates[0]
            action = actions[0]
            if not _apply_action_strict(engine, prob, loc, term, action):
                break
            continue

        _, loc, term = target_terms[0]
        if mode == "sdp":
            if _try_action(engine, prob, loc, term, "sdp_relaxation"):
                continue
            if not _try_action(engine, prob, loc, term, "mccormick_relaxation"):
                break
        elif mode == "structure":
            action = choose_structure_action(term, k_min=k_min, tau_density=tau_density)
            if _apply_action_strict(engine, prob, loc, term, action):
                continue
            break
        else:
            if not _try_action(engine, prob, loc, term, "mccormick_relaxation"):
                break

    return prob
