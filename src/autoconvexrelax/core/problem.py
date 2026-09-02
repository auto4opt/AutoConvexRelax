from collections import Counter
import copy
import sympy
from sympy import *
from typing import List, Dict, Tuple, Set, Optional, Union
from sympy.matrices.expressions.matexpr import MatrixElement

def normalize_scalar(expr):
    """
    只在 *稠密* MatrixBase 且 1×1 时取 [0,0]；MatrixExpr (MatMul/Trace/Transpose...) 不动。
    """
    if isinstance(expr, sympy.MatrixBase) and getattr(expr, "shape", None) == (1, 1):
        return expr[0, 0]    # 仅限稠密矩阵
    return expr              # MatrixExpr 原样保留


class Variable:
    """
    表示一个优化变量的类。
    可扩展支持更多属性，如:
    - 'vtype': 'continuous'/'integer'/'binary'
    - 'initial_value': 初始解(用于启发式或warm start)
    """

    def __init__(self, name: str,
                 lb: Optional[float] = None,
                 ub: Optional[float] = None,
                 vtype: str = 'continuous' # 'continuous'/'integer'/'binary'/'matrix'
                 ):
        self.name = name
        self.lb = lb
        self.ub = ub
        self.vtype = vtype
        self.sym = None  # sympy.Symbol 句柄（用于全局复用，避免同名不同句柄）


    def __repr__(self):
        return f"Variable(name={self.name}, lb={self.lb}, ub={self.ub}, vtype={self.vtype})"
    
class MatrixVariableSymbol:
    """
    表示一个整体的矩阵变量，比如 X ∈ ℝ^{m×n}。
    用 MatrixSymbol 表示，便于进行高阶表达式构造（如 trace, PSD）。
    """
    def __init__(self, name: str, rows: int, cols: int, vtype='continuous', lb=None, ub=None):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.symbol = MatrixSymbol(name, rows, cols)
        self.vtype = vtype  # continuous / symmetric / etc
        self.lb = lb
        self.ub = ub

    def __repr__(self):
        return f"MatrixVariableSymbol(name={self.name}, shape={self.rows}x{self.cols})"


class VectorVariableSymbol:
    """
    列向量变量 x ∈ ℝ^n
    """
    def __init__(self, name: str, dim: int, vtype='continuous', lb=None, ub=None):
        self.name  = name
        self.dim   = dim
        self.vtype = vtype
        self.lb    = lb
        self.ub    = ub
        # 本质仍用 MatrixSymbol，shape=(n,1)
        self.symbol = MatrixSymbol(name, dim, 1)

    # 让 x[i] 语法成立
    def __getitem__(self, idx):
        if not 0 <= idx < self.dim:
            raise IndexError("index out of range")
        return self.symbol[idx, 0]

    def __repr__(self):
        return f"VectorVariableSymbol(name={self.name}, dim={self.dim}, vtype={self.vtype}, lb={self.lb}, ub={self.ub})"


class Constraint:
    """
    表示单个约束，如 expr <= rhs, expr >= rhs, expr == rhs 等
    其中 expr 和 rhs 都可以是 sympy 表达式，也可以是float（对 rhs 通常是常数）
    """

    def __init__(self,
                 expr: sympy.Expr,
                 sense: str,
                 rhs: Union[sympy.Expr, float, int]):
        """
        :param expr: 左侧表达式
        :param sense: in {'<=','>=','='}
        :param rhs: 右侧，可以是float/int或sympy表达式
        """
        if sense not in ['<=', '>=', '=', '<', '>', 'is']:
            raise ValueError("sense must be one of <=, >=, =, <, >, is.")
        self.expr = expr
        self.sense = sense
        self.rhs = rhs

    def __repr__(self):
        return f"Constraint({self.expr} {self.sense} {self.rhs})"
    
    def __str__(self):
        return f"{self.expr} {self.sense} {self.rhs}"


class PSDConstraint:
    """
    表示一个 PSD 约束: M(x) >= 0 (in PSD sense).
    其中 M(x) 通常是一个对称矩阵，依赖于问题变量。
    """

    def __init__(self, matrix_expr):
        # matrix_expr 是个 Sympy Matrix, 里面含有 x1, x2, ... 等符号
        self.matrix_expr = matrix_expr

    def __repr__(self):
        return f"M = ({self.matrix_expr}) is positive semidefinite"


class QCQPProblem:
    """
    存放一个QCQP问题的所有信息:
    - 变量集
    - 目标函数 (obj_expr)
    - 约束列表
    - 目标类型 (minimize / maximize)
    """

    def __init__(self, name="QCQP_Problem", sense='min'):
        self.name = name
        self.variables = {}  # {var_name: Variable}
        self.obj_expr = None  # Sympy 表达式
        self.obj_sense = sense  # 或 'max'
        self.constraints: List[Constraint] = []
        self.psd_constraints: List[PSDConstraint] = []
        self.item_to_id = {}  # (表达式项, 所在位置) -> 编号
        self.id_to_item = {}  # 编号 -> (表达式项, 所在位置)
        self.items = []  # 所有项
        self.counter = 0  # 从1开始编号
        self.matrix_variables = {}  # {var_name: MatrixVariableSymbol}
        # [FASTPATH] 记录 SDP/SDR 引入的 lifting 矩阵变量名（如 Z_x, Z_x_1, Z_T_xxx ...）
        self._sdp_Z_names = set()
        self._me_bounds = {}  # key: (base:str, i:int, j:int) -> (lb:float|None, ub:float|None)


    # def add_variable(self, var: Variable):
    #     self.variables[var.name] = var
    def _me_key(self, base, i, j):
        return (str(base), int(i), int(j))

    def get_me_bounds(self, base, i, j):
        if not hasattr(self, "_me_bounds") or self._me_bounds is None:
            self._me_bounds = {}
        return self._me_bounds.get(self._me_key(base, i, j), (None, None))


    def tighten_me_bounds(self, base, i, j, new_lb=None, new_ub=None, tol=1e-9):
        import math
        if not hasattr(self, "_me_bounds") or self._me_bounds is None:
            self._me_bounds = {}
        k = self._me_key(base, i, j)
        cur_lb, cur_ub = self._me_bounds.get(k, (None, None))

        def norm(x):
            if x is None: return None
            x = float(x)
            if math.isnan(x) or math.isinf(x): return None
            return x

        new_lb, new_ub = norm(new_lb), norm(new_ub)
        cur_lb = None if cur_lb is None else float(cur_lb)
        cur_ub = None if cur_ub is None else float(cur_ub)

        prop_lb, prop_ub = cur_lb, cur_ub
        if new_lb is not None:
            if prop_lb is None or new_lb > prop_lb + tol:
                prop_lb = new_lb
        if new_ub is not None:
            if prop_ub is None or new_ub < prop_ub - tol:
                prop_ub = new_ub

        # guard
        if prop_lb is not None and prop_ub is not None and prop_lb > prop_ub + tol:
            return False

        self._me_bounds[k] = (prop_lb, prop_ub)
        return (prop_lb != cur_lb) or (prop_ub != cur_ub)

    def _expand_trace_linearity(self, expr):
        """
        仅对 Trace 做线性展开，并把 Trace(α*M) 的 α 提到外层（变成 α*Trace(M)）。
        不做其它 simplify，避免 SymPy 合并多条 Trace。
        """
        import sympy as sp

        def is_matlike(a):
            return isinstance(a, (sp.MatrixExpr, sp.MatrixBase))

        def rec(e):
            if isinstance(e, sp.Trace):
                inner = rec(e.arg)

                # Trace(A + B + ...) → Trace(A)+Trace(B)+...
                if isinstance(inner, (sp.Add, sp.MatAdd)):
                    return sp.Add(*[sp.Trace(rec(a)) for a in inner.args])

                # Trace(α * M(...))：把标量 α 提出来，保留为 α*Trace(core)
                if isinstance(inner, sp.Mul):
                    mats = [a for a in inner.args if is_matlike(a)]
                    scas = [a for a in inner.args if not is_matlike(a)]
                    if scas and mats:
                        alpha = sp.Mul(*scas)
                        core  = mats[0] if len(mats)==1 else sp.MatMul(*mats)
                        return alpha * sp.Trace(rec(core))

                return sp.Trace(inner)

            # 结构递归，避免触发全局化简
            if isinstance(e, (sp.Add, sp.MatAdd, sp.Mul, sp.MatMul, sp.Function, sp.Pow, sp.Transpose)):
                return e.func(*(rec(a) for a in e.args))
            return e

        return rec(expr)
    
    # ==== 新增：统一的预处理括号展开 ====
    def _pre_expand(self, expr):
        """
        轻量级括号展开：
        - 标量表达式: 用 sympy.expand(mul=True) 展开乘法 (a+b)*c → a*c + b*c
        - 矩阵/向量表达式: 只对 MatMul 里的加法做分配律展开：
              (A + B)*C      → A*C + B*C
              C*(A + B)      → C*A + C*B
              A*(B + C)*D    → A*B*D + A*C*D
          不会把向量拆成标量分量。
        """
        import sympy as sp

        # ------- 针对矩阵/向量的分配律展开 -------
        def distribute_matmul(e):
            # 递归处理加法内部
            if isinstance(e, (sp.Add, sp.MatAdd)):
                return e.func(*(distribute_matmul(a) for a in e.args))

            # Trace 里也可能有 (A+B)*X 这类结构，递归进去
            if isinstance(e, sp.Trace):
                return sp.Trace(distribute_matmul(e.arg))

            # 只在 MatMul/Mul 上对 Add 做分配律
            if isinstance(e, (sp.Mul, sp.MatMul)):
                # 先递归处理每个因子
                args = [distribute_matmul(a) for a in e.args]

                # 找到第一个 Add/MatAdd 因子，按它来拆
                add_indices = [i for i, a in enumerate(args)
                               if isinstance(a, (sp.Add, sp.MatAdd))]
                if not add_indices:
                    # 没有加法因子，直接还原
                    return e.func(*args)

                i = add_indices[0]
                add_arg = args[i]

                # ( ... * (A+B) * ... ) → Σ_t (... * t * ...)
                terms = []
                for t in add_arg.args:
                    new_args = list(args)
                    new_args[i] = t
                    terms.append(distribute_matmul(e.func(*new_args)))
                return sp.Add(*terms)

            # 对转置、幂等其它结构，递归进去但不做分配
            if isinstance(e, (sp.Transpose, sp.Inverse, sp.Adjoint, sp.Function, sp.Pow)):
                return e.func(*(distribute_matmul(a) for a in e.args))

            # 其它类型保持不变
            return e

        # ------- 先把 1×1 稠密矩阵当成标量处理 -------
        if isinstance(expr, sp.MatrixBase) and getattr(expr, "shape", None) == (1, 1):
            expr = expr[0, 0]

        # ------- MatrixExpr 走“矩阵分配律”路径 -------
        if isinstance(expr, sp.MatrixExpr):
            return distribute_matmul(expr)

        # ------- 纯标量表达式用 sympy.expand 展开乘法 -------
        expr = sp.expand(expr, mul=True,
                         power_exp=False, power_base=False, log=False)
        return expr

    
    def _split_or_add(self, expr, location):
        import sympy as sp

        # --- [PATCH] unwrap 1x1 scalar matrix wrapper ---
        if isinstance(expr, sp.MatrixBase) and getattr(expr, "shape", None) == (1, 1):
            expr = expr[0, 0]
            
        def is_matlike(a):
            return isinstance(a, (sp.MatrixExpr, sp.MatrixBase))

        # 0) 顶层加法：直接拆成若干项，每一项自己带着自己的系数
        if isinstance(expr, (sp.Add, sp.MatAdd)):
            for term in expr.args:
                self._split_or_add(term, location)
            return

        # 1) Trace：这里不再展开，展开已经在 map_*_terms 里统一做过
        #    比如 Trace(A+B) 会在 _expand_trace_linearity 里变成 Trace(A)+Trace(B)
        if isinstance(expr, sp.Trace):
            # 这里 expr 可能是纯 Trace(...)，也可能是已经被 _expand_trace_linearity 处理过的形式
            self._add_term(expr, location)
            return

        # 2) α*Trace(core)：保留完整的 α*Trace(core)，包括符号和系数
        #    比如 0.5*Trace(x.T*Q*x)、-Trace(y.T*Q*x)
        if isinstance(expr, (sp.Mul, sp.MatMul)) and any(isinstance(a, sp.Trace) for a in expr.args):
            traces = [a for a in expr.args if isinstance(a, sp.Trace)]
            others = [a for a in expr.args if not isinstance(a, sp.Trace)]
            # 只处理“一个 Trace + 若干纯标量”的情况
            if len(traces) == 1 and all(not is_matlike(o) for o in others):
                # 关键：直接把完整 expr 存进去，不再剥掉标量
                self._add_term(expr, location)
                return

        # 3) 其它含矩阵因子的乘积：保留完整 expr（含符号和系数），再交给 _add_term 处理
        #    例如：-x.T*Q*x, 2*(x.T*x), 3*A*X*B 等
        if isinstance(expr, (sp.Mul, sp.MatMul)) and any(is_matlike(a) for a in expr.args):
            self._add_term(expr, location)
            return

        # 4) 其它 MatrixExpr（Transpose/MatMul/MatrixSymbol...）整体入库
        if isinstance(expr, sp.MatrixExpr):
            self._add_term(expr, location)
            return

        # 5) 纯标量：按加法最细拆（常数在 _add_term 里会被过滤掉）
        for term in sp.Add.make_args(expr):
            self._add_term(term, location)


    def _is_psd_numeric(self, Q, tol=1e-8):
        """
        数值方式检查 Q 是否 PSD：所有特征值 >= -tol 则视为 PSD。
        仅在 Q 不含符号变量时使用。
        返回 True / False / None (异常时).
        """
        import numpy as np
        import sympy as sp
        try:
            # 统一成 Matrix
            if isinstance(Q, sp.MatrixExpr):
                Qm = sp.Matrix(Q)
            else:
                Qm = sp.Matrix(Q)

            # 如果还带有自由符号，就不要做数值判定
            if Qm.free_symbols:
                return None

            Q_num = np.array(Qm.evalf(), dtype=float)
            vals  = np.linalg.eigvalsh(Q_num)
            return float(vals.min()) >= -tol
        except Exception:
            return None

        
    def add_variable(self, name: str, lb=None, ub=None, vtype='continuous'):
        if name in self.variables:
            raise ValueError(f"Variable {name} already exists.")
        var = Variable(name=name, lb=lb, ub=ub, vtype=vtype)
        self.variables[name] = var
        sym = sympy.Symbol(name, real=True)
        var.sym = sym
        return sym
    
    def add_vector_variable(self, name: str, dim: int, vtype='continuous', lb=None, ub=None):
        if name in self.matrix_variables:
            raise ValueError(f"Variable {name} already exists.")
        vec = VectorVariableSymbol(name, dim, vtype, lb, ub)
        self.matrix_variables[name] = vec  # 依旧复用字典
        return vec.symbol                   # 让用户直接拿到 SymPy 对象

    
    def add_matrix_variable(self, name: str, rows: int, cols: int, vtype='continuous', lb=None, ub=None):
        if name in self.matrix_variables:
            raise ValueError(f"Matrix variable {name} already exists.")
        matrix_var = MatrixVariableSymbol(name, rows, cols, vtype, lb, ub)
        self.matrix_variables[name] = matrix_var
        return matrix_var.symbol
        
    def get_matrix_symbol(self, name: str):
        if name not in self.matrix_variables:
            raise KeyError(f"Matrix variable {name} not found.")
        return self.matrix_variables[name].symbol

    def set_objective(self, expr, sense='min'):
        """设置目标函数表达式"""
        if not isinstance(expr, (sympy.Expr, sympy.MatrixBase, sympy.MatrixExpr)):
            raise TypeError("Objective must be Expr / MatrixBase / MatrixExpr.")

        # self.obj_expr = expr
        self.obj_expr  = expr
        self.obj_sense = sense

    def add_constraint(self, expr: sympy.Expr, sense: str, rhs: Union[sympy.Expr, float, int, str]):
        """向问题添加约束"""
        # expr = normalize_scalar(expr)
        # if isinstance(rhs, sympy.Expr) or isinstance(rhs, sympy.MatrixBase) or isinstance(rhs, sympy.MatrixExpr):
        #     rhs = normalize_scalar(rhs)
        c = Constraint(expr, sense, rhs)
        self.constraints.append(c)
        
    def add_constraint_unique(self, expr, sense, rhs):
        import sympy as sp
        if not hasattr(self, "_seen_cons"):
            self._seen_cons = set()

        # 不要 simplify：它会把每条约束都变成一次“全局化简任务”
        # 用结构表达式做 key，稳定且快
        expr_key = sp.srepr(expr)
        rhs_key  = sp.srepr(rhs) if isinstance(rhs, (sp.Expr, sp.MatrixExpr, sp.MatrixBase)) else str(rhs)
        key = (expr_key, sense, rhs_key)

        if key in self._seen_cons:
            return
        self._seen_cons.add(key)
        self.add_constraint(expr, sense, rhs)


        
    def add_psd_constraint(self, matrix_expr):
        """
        向问题中添加一个 PSD 约束，确保 matrix_expr 是对称矩阵。
        """
        if not isinstance(matrix_expr, sympy.MatrixExpr):
            raise TypeError("PSD constraint must be a sympy.Matrix.")

        self.psd_constraints.append(PSDConstraint(matrix_expr))

        # [FASTPATH] 从 PSD 约束里识别 SDP 新引入的 Z（通常名字以 'Z' 开头）
        try:
            # 只考虑“已注册的矩阵决策变量”
            registered = {
                mv.symbol.name
                for mv in getattr(self, "matrix_variables", {}).values()
                if hasattr(mv, "symbol") and isinstance(mv.symbol, sympy.MatrixSymbol)
            }
            for ms in matrix_expr.atoms(sympy.MatrixSymbol):
                n = getattr(ms, "name", None)
                if n in registered and isinstance(n, str) and n.startswith("Z"):
                    self._sdp_Z_names.add(n)
        except Exception:
            # 纯加速用的缓存，失败不影响正确性
            pass

        # [FASTPATH] term 里是否出现 SDP 引入的 Z（或 Z_ij 这种 MatrixElement）
    def _term_involves_sdp_Z(self, expr: sympy.Expr) -> bool:
        znames = getattr(self, "_sdp_Z_names", set())
        if not znames:
            return False

        # 1) 直接出现 MatrixSymbol: ... * Z * ...
        try:
            for ms in expr.atoms(sympy.MatrixSymbol):
                if getattr(ms, "name", None) in znames:
                    return True
        except Exception:
            pass

        # 2) 出现 MatrixElement: Z[i,j]
        try:
            from sympy.matrices.expressions.matexpr import MatrixElement
            for me in expr.atoms(MatrixElement):
                parent = getattr(me, "parent", None) or getattr(me, "base", None) or me.args[0]
                pname = getattr(parent, "name", getattr(parent, "label", None))
                if pname in znames:
                    return True
        except Exception:
            pass

        return False


    def get_sympy_symbol(self, var_name: str) -> sympy.Symbol:
        """
        给定一个变量名, 返回对应的 Sympy Symbol.
        如果不存在, 这里自动创建(也可改成抛错).
        """
        if var_name not in self.variables:
            raise KeyError(f"Variable {var_name} not found in problem.")
        return Symbol(var_name, real=True)
    
    def _is_decision_vector_matrix_symbol(self, term):
        """
        返回 term 是否是“决策向量/矩阵变量本体”（或其转置）。
        仅把整体变量本体视为需要跳过的叶子；含有运算（如 x.T*x、Trace(...)) 的表达式仍应保留。
        """
        import sympy as sp
        # 收集所有已注册的 MatrixSymbol
        mat_syms = set()
        for mv in getattr(self, "matrix_variables", {}).values():
            if hasattr(mv, "symbol"):
                mat_syms.add(mv.symbol)

        # 1) 直接是 MatrixSymbol 本体
        if isinstance(term, sp.MatrixSymbol) and term in mat_syms:
            return True

        # 2) 转置后的 MatrixSymbol 本体
        if isinstance(term, sp.Transpose) and isinstance(term.arg, sp.MatrixSymbol) and term.arg in mat_syms:
            return True

        return False
    
    # 在 QCQPProblem._add_term 的最开头，加入“跳过裸向量/矩阵”的防线
    def _add_term(self, term, location: str):
        import sympy as sp

        # ---- 0) 跳过“决策向量/矩阵本体（含 .T）” ----
        if self._is_decision_vector_matrix_symbol(term):
            return

        # ---- 1) 如果是顶层加法，继续拆 ----
        if isinstance(term, (sp.Add, sp.MatAdd)):
            for t in term.args:
                self._add_term(t, location)
            return

        # ---- 2) 常数直接忽略（与图侧一致）----
        is_constant_term = False
        if hasattr(term, 'is_Number') and term.is_Number:
            is_constant_term = True
        elif isinstance(term, sp.MatrixBase) and not term.free_symbols:
            is_constant_term = True
        if is_constant_term:
            return

        # ---- 3) MatMul/Mul：若只有一个矩阵因子且该因子是“决策向量/矩阵本体”，也跳过 ----
        if isinstance(term, (sp.Mul, sp.MatMul)):
            mats = [a for a in term.args if isinstance(a, (sp.MatrixExpr, sp.MatrixBase))]
            others = [a for a in term.args if a not in mats]
            if len(mats) == 1 and self._is_decision_vector_matrix_symbol(mats[0]) and all(not isinstance(o, (sp.MatrixExpr, sp.MatrixBase)) for o in others):
                # 例如 α * x 这种形式（x 是决策向量本体）→ 跳过
                return

        # ---- 4) 正常入库 ----
        key = (sp.srepr(term), location)
        if key not in self.item_to_id:
            self.counter += 1
            self.item_to_id[key] = self.counter
            self.id_to_item[self.counter] = (term, location)
            

    # ------------------------------------------------------------------
    # Preprocessing: canonicalize quadratic/bilinear matrix forms
    # ------------------------------------------------------------------
    
    def _canonicalize_quad_bilin(self, expr):
        """Canonicalize scalar quadratic/bilinear matrix forms.

        This is purely algebraic (no relaxation yet) and helps:
        1) Merge duplicate terms like x^T Q y and y^T Q^T x into a single canonical form;
        2) Symmetrize quadratic forms x^T Q x via (Q+Q^T)/2, which is the standard practice
           before SDP-based relaxations.

        Canonical rule: order variables by lexicographic name (dictionary order).
        """
        import sympy as sp
        from sympy.matrices.expressions import MatMul, Transpose, Trace

        def canon_matmul(m):
            # Only handle scalar (1x1) matrix products
            try:
                if getattr(m, "shape", None) != (1, 1):
                    return m
            except Exception:
                return m

            if not isinstance(m, MatMul):
                return m

            args = list(m.args)
            # Pull out leading scalar coefficient if present
            coeff = sp.Integer(1)
            if args and getattr(args[0], "is_number", False):
                coeff = args[0]
                args = args[1:]

            # Pattern: x.T * Q * y
            if len(args) == 3 and isinstance(args[0], Transpose):
                x = args[0].args[0]
                Q = args[1]
                y = args[2]
                if not (hasattr(x, "name") and hasattr(y, "name")):
                    return m

                xn, yn = x.name, y.name

                if xn == yn:
                    Qsym = sp.Rational(1, 2) * (Q + Q.T)
                    return coeff * (x.T * Qsym * x)

                if xn <= yn:
                    return coeff * (x.T * Q * y)
                else:
                    # y.T*Q*x -> x.T*Q.T*y (after swap)
                    return coeff * (y.T * Q.T * x)

            return m

        def transform(e):
            if isinstance(e, Trace):
                inner = e.args[0]
                if isinstance(inner, MatMul):
                    return Trace(canon_matmul(inner))
                return e
            if isinstance(e, MatMul):
                return canon_matmul(e)
            return e

        # Apply recursively; keep it light (no global simplify here)
        return expr.replace(lambda e: isinstance(e, (Trace, MatMul)), transform)


    def _canonicalize_constraint_senses_inplace(self):
        """
        Normalize constraints to '<=' form in-place.

        - f(x) >= rhs  ->  -f(x) <= -rhs
        - f(x) >  rhs  ->  -f(x) <= -rhs   (treated as non-strict)
        - f(x) <  rhs  ->   f(x) <=  rhs   (treated as non-strict)

        - f(x) = rhs   ->   f(x) <= rhs    AND    -f(x) <= -rhs
        (This makes the whole pipeline only need to handle '<=' constraints.)
        """
        import sympy as sp

        new_constraints = []

        for cons in getattr(self, "constraints", []):
            if not hasattr(cons, "sense"):
                new_constraints.append(cons)
                continue

            s = str(cons.sense).strip()

            # normalize strict to non-strict (optional, but practical)
            if s == "<":
                cons.sense = "<="
                new_constraints.append(cons)
                continue

            if s == ">":
                s = ">="

            # normalize equality aliases
            if s == "==":
                s = "="

            # helper: robust rhs sympify (so -rhs works for "1.0"/etc.)
            rhs = cons.rhs
            if isinstance(rhs, str):
                try:
                    rhs = sp.sympify(rhs)
                except Exception:
                    pass

            if s == ">=":
                # flip both sides: f >= rhs  ->  -f <= -rhs
                cons.expr = -cons.expr
                try:
                    cons.rhs = -rhs
                except Exception:
                    cons.rhs = rhs
                cons.sense = "<="
                new_constraints.append(cons)
                continue

            if s == "=":
                # equality -> two inequalities
                # 1) f <= rhs
                c1 = Constraint(cons.expr, "<=", rhs)

                # 2) -f <= -rhs
                try:
                    rhs2 = -rhs
                except Exception:
                    rhs2 = rhs
                c2 = Constraint(-cons.expr, "<=", rhs2)

                new_constraints.append(c1)
                new_constraints.append(c2)
                continue

            # already <= (or other senses you may have)
            if s == "<=":
                cons.sense = "<="
            new_constraints.append(cons)

        self.constraints = new_constraints

    # ------------------------------------------------------------
    # One-time fixed-variable preprocessing (run before term mapping)
    # ------------------------------------------------------------
    def _elem_bounds(self, mv, idx):
        """
        支持 mv.lb/mv.ub 为：
        - scalar
        - 1D list/np
        - 2D list/np
        """
        import numbers
        lb = getattr(mv, "lb", None)
        ub = getattr(mv, "ub", None)
        if lb is None or ub is None:
            return None, None
        if isinstance(lb, numbers.Number) and isinstance(ub, numbers.Number):
            return float(lb), float(ub)
        if isinstance(idx, tuple):
            i, j = idx
            try:
                return float(lb[i][j]), float(ub[i][j])
            except Exception:
                return None, None
        try:
            return float(lb[idx]), float(ub[idx])
        except Exception:
            return None, None

    def preprocess_fixed_vars(self) -> bool:
        """
        One-time preprocessing (run before map_all_terms):
        - Fix scalar vars with lb == ub
        - Detect vector sum constraints like sum(y) = k and fix all elements when implied
        Returns True if substitutions were applied.
        """
        import sympy as sp

        if getattr(self, "_fixed_var_preprocessed", False):
            return False

        subs = {}

        # 1) scalar fixed vars
        for name, v in getattr(self, "variables", {}).items():
            lb = getattr(v, "lb", None)
            ub = getattr(v, "ub", None)
            if lb is None or ub is None:
                continue
            if lb == ub:
                subs[sp.Symbol(name)] = float(lb)

        # 2) vector sum equality -> fix elements when implied
        def _linear_coeffs(expr):
            atoms = list(expr.atoms(sp.Symbol, MatrixElement))
            if not atoms:
                return None
            repl = {}
            sym_map = {}
            for i, a in enumerate(atoms):
                s = sp.Symbol(f"__bt_dummy_{i}")
                repl[a] = s
                sym_map[s] = a
            expr2 = expr.xreplace(repl)
            try:
                poly = sp.Poly(expr2, *sym_map.keys(), domain="RR")
            except Exception:
                return None
            if poly.total_degree() > 1:
                return None
            d = poly.as_dict()
            const = float(d.get((0,) * len(sym_map), 0.0))
            coeffs = {}
            for exp, coef in d.items():
                if sum(exp) == 0:
                    continue
                if sum(exp) == 1:
                    idx = exp.index(1)
                    coeffs[sym_map[list(sym_map.keys())[idx]]] = float(coef)
                else:
                    return None
            return coeffs, const

        def _expand_trace(tr):
            inner = tr.arg
            # Trace(1x1) -> scalar
            if isinstance(inner, sp.MatrixExpr) and getattr(inner, "shape", None) == (1, 1):
                return inner[0, 0]
            # Trace(Z) where Z is square MatrixSymbol -> sum_i Z[i,i]
            if isinstance(inner, sp.MatrixSymbol) and inner.shape[0] == inner.shape[1]:
                n = int(inner.shape[0])
                return sp.Add(*[inner[i, i] for i in range(n)])
            # Trace(row * y)
            if isinstance(inner, sp.MatMul) and len(inner.args) == 2:
                A, B = inner.args
                if isinstance(A, (sp.MatrixBase, sp.ImmutableMatrix)) and isinstance(B, sp.MatrixSymbol):
                    if A.shape[0] == 1 and B.shape[1] == 1 and A.shape[1] == B.shape[0]:
                        n = int(B.shape[0])
                        return sp.Add(*[A[0, i] * B[i, 0] for i in range(n)])
            return tr

        def _expand_sum(s):
            try:
                if not isinstance(s, sp.Sum):
                    return s
                (sym, start, end) = s.limits[0]
                start = int(start)
                end = int(end)
                terms = [s.function.subs(sym, i) for i in range(start, end + 1)]
                return sp.Add(*terms)
            except Exception:
                return s

        sum_eq = []  # list of (parent, k, sense)

        # direct Trace(row*y) detection
        for c in getattr(self, "constraints", []):
            expr = c.expr
            rhs = c.rhs
            sense = c.sense
            if sense in [">", ">="]:
                expr = -expr
                rhs = -rhs
                sense = "<="
            elif sense in ["=", "=="]:
                sense = "="

            try:
                rhs_val = float(rhs)
            except Exception:
                continue

            if isinstance(expr, sp.Trace):
                inner = expr.arg
                if isinstance(inner, sp.MatMul) and len(inner.args) == 2:
                    A, B = inner.args
                    if isinstance(A, (sp.MatrixBase, sp.ImmutableMatrix)) and isinstance(B, sp.MatrixSymbol):
                        if A.shape[0] == 1 and B.shape[1] == 1 and A.shape[1] == B.shape[0]:
                            try:
                                if all(float(A[0, i]) == 1.0 for i in range(A.shape[1])):
                                    if sense == "=":
                                        sum_eq.append((B, rhs_val, "<="))
                                        sum_eq.append((B, rhs_val, ">="))
                                    else:
                                        sum_eq.append((B, rhs_val, sense))
                            except Exception:
                                pass

        # linear scan
        for c in getattr(self, "constraints", []):
            expr = c.expr
            rhs = c.rhs
            try:
                expr = expr.replace(lambda x: isinstance(x, sp.Trace), _expand_trace)
                expr = expr.replace(lambda x: isinstance(x, sp.Sum), _expand_sum)
            except Exception:
                pass

            if c.sense in [">", ">="]:
                expr = -expr
                rhs = -rhs
            elif c.sense in ["=", "=="]:
                pass

            try:
                rhs_val = float(rhs)
            except Exception:
                continue

            senses = ["<="] if c.sense not in ["=", "=="] else ["<=", ">="]
            for s in senses:
                e = expr if s == "<=" else -expr
                r = rhs_val if s == "<=" else -rhs_val
                try:
                    e = sp.expand(e)
                except Exception:
                    pass

                lin = _linear_coeffs(e)
                if lin is None:
                    continue
                coeffs, const = lin
                if not coeffs:
                    continue

                parents = []
                for a in coeffs.keys():
                    if not isinstance(a, MatrixElement):
                        parents = []
                        break
                    parent = getattr(a, "parent", None) or getattr(a, "base", None)
                    if not isinstance(parent, sp.MatrixSymbol):
                        parents = []
                        break
                    parents.append(parent)
                if not parents:
                    continue
                parent = parents[0]
                if any(p != parent for p in parents):
                    continue
                coeff_vals = list(coeffs.values())
                if max(coeff_vals) - min(coeff_vals) > 1e-9:
                    continue
                cval = coeff_vals[0]
                if abs(cval) < 1e-12:
                    continue
                k = (r - const) / cval
                if cval < 0:
                    s = ">=" if s == "<=" else "<="
                sum_eq.append((parent, k, s))

        eq_map = {}
        for parent, k, s in sum_eq:
            key = (parent, round(float(k), 9))
            eq_map.setdefault(key, set()).add(s)

        for (parent, k), senses in eq_map.items():
            if "<=" not in senses or ">=" not in senses:
                continue
            mv = getattr(self, "matrix_variables", {}).get(parent.name)
            if mv is None:
                continue
            n = int(parent.shape[0])
            if getattr(parent, "shape", None) != (n, 1):
                continue
            lbs = []
            ubs = []
            for i in range(n):
                li, ui = self._elem_bounds(mv, i)
                if li is None or ui is None:
                    lbs = []
                    break
                lbs.append(li)
                ubs.append(ui)
            if not lbs:
                continue
            if max(lbs) - min(lbs) > 1e-9 or max(ubs) - min(ubs) > 1e-9:
                continue
            l = lbs[0]
            u = ubs[0]
            if abs(k - n * u) < 1e-6:
                val = u
            elif abs(k - n * l) < 1e-6:
                val = l
            else:
                continue
            for i in range(n):
                subs[MatrixElement(parent, i, 0)] = float(val)

        # promote fully fixed MatrixSymbols to constant matrices
        fixed_parents = {}
        for sym, val in subs.items():
            if isinstance(sym, MatrixElement):
                parent = getattr(sym, "parent", None) or getattr(sym, "base", None)
                if parent is None:
                    continue
                try:
                    i = int(sym.i)
                    j = int(sym.j)
                except Exception:
                    continue
                fixed_parents.setdefault(parent, {})[(i, j)] = float(val)

        for parent, elems in fixed_parents.items():
            try:
                rows, cols = int(parent.shape[0]), int(parent.shape[1])
            except Exception:
                continue
            if len(elems) != rows * cols:
                continue
            complete = True
            for i in range(rows):
                for j in range(cols):
                    if (i, j) not in elems:
                        complete = False
                        break
                if not complete:
                    break
            if not complete:
                continue
            mat = sp.Matrix(rows, cols, lambda i, j: elems[(i, j)])
            subs[parent] = mat

        if not subs:
            self._fixed_var_preprocessed = True
            return False

        try:
            fixed_preview = list(subs.items())[:6]
            if getattr(self, "debug_fix", False):
                print(f"[FIXED] substitutions={len(subs)} preview={fixed_preview}")
        except Exception:
            pass

        try:
            self.obj_expr = sp.simplify(sp.expand(self.obj_expr.xreplace(subs)))
        except Exception:
            self.obj_expr = self.obj_expr.xreplace(subs)

        for c in getattr(self, "constraints", []):
            try:
                c.expr = sp.simplify(sp.expand(c.expr.xreplace(subs)))
            except Exception:
                c.expr = c.expr.xreplace(subs)

        for sym, val in subs.items():
            if isinstance(sym, sp.Symbol):
                v = self.variables.get(sym.name)
                if v is not None:
                    v.lb = val
                    v.ub = val

        self._fixed_var_preprocessed = True
        return True

    # def preprocess(self, normalize_quad_bilin: bool = True, simplify: bool = True, rebuild_term_map: bool = True):
    #     """In-place preprocessing for consistent term identities across objective/constraints.

    #     Typical usage: call once after building the problem (objective+constraints),
    #     and before RL/term-mapping/relaxations.
    #     """
    #     import sympy as sp
    #     if normalize_quad_bilin:
    #         self.obj_expr = self._canonicalize_quad_bilin(self.obj_expr)
    #         for c in self.constraints:
    #             c.expr = self._canonicalize_quad_bilin(c.expr)

    #     if simplify:
    #         try:
    #             self.obj_expr = sp.simplify(self.obj_expr)
    #         except Exception:
    #             pass
    #         for c in self.constraints:
    #             try:
    #                 c.expr = sp.simplify(c.expr)
    #             except Exception:
    #                 pass

    #     if rebuild_term_map:
    #         # Rebuild term ↔ id mapping so duplicated forms merge into one term.
    #         self.item_to_id.clear()
    #         self.id_to_item.clear()
    #         self._next_id = 0
    #         self.map_objective_terms(update_problem=False)
    #         for i in range(len(self.constraints)):
    #             self.map_constraint_terms(i, update_problem=False)

    def map_objective_terms(self, update_problem: bool = False):
        """
        把目标函数拆成若干 term 做映射。
        若 update_problem=True，则会把预处理后的 expr 回写到 self.obj_expr，
        保证后续在原问题上找到相同的子表达式。
        """
        if self.obj_expr is None:
            raise ValueError("Objective function not set.")

        # 先做一次轻量括号展开（只针对标量）
        expr = self._pre_expand(self.obj_expr)
        # 再做 Trace 线性展开（拆成多个 Trace）
        expr = self._expand_trace_linearity(expr)

        expr = self._canonicalize_quad_bilin(expr)

        # 关键：根据需要把预处理结果写回到问题本体
        if update_problem:
            self.obj_expr = expr

        # 最后按加法拆成 term
        self._split_or_add(expr, "Objective")


    def map_constraint_terms(self, update_problem: bool = False):
        import sympy as sp
        for i, cons in enumerate(self.constraints):
            lhs_loc = f"Constraint_{i+1}_LHS"
            rhs_loc = f"Constraint_{i+1}_RHS"

            # 先对 lhs 做括号展开（只动标量），再做 Trace 线性展开
            lhs = self._pre_expand(cons.expr)
            lhs = self._expand_trace_linearity(lhs)
            lhs = self._canonicalize_quad_bilin(lhs)

            # 写回到原 problem 里
            if update_problem:
                cons.expr = lhs

            self._split_or_add(lhs, lhs_loc)

            rhs = cons.rhs
            if isinstance(rhs, (sp.Expr, sp.MatrixBase, sp.MatrixExpr)):
                rhs = self._pre_expand(rhs)
                rhs = self._expand_trace_linearity(rhs)
                rhs = self._canonicalize_quad_bilin(rhs)

                # 同样写回
                if update_problem:
                    cons.rhs = rhs

                self._split_or_add(rhs, rhs_loc)


    def map_all_terms(self, update_problem: bool = True):
        """
        映射目标函数和所有约束的所有项。
        若 update_problem=True，则在映射前会把预处理（expand/Trace 展开）后的表达式
        回写到 obj_expr / constraints[*].expr / rhs 中。
        """
        self.item_to_id.clear()
        self.id_to_item.clear()
        self.items.clear()
        self.counter = 0
        
        if update_problem:
            self._canonicalize_constraint_senses_inplace()
            # fixed-variable preprocessing before mapping
            try:
                self.preprocess_fixed_vars()
            except Exception:
                pass
        
        self.map_objective_terms(update_problem=update_problem)
        self.map_constraint_terms(update_problem=update_problem)

        # ---- convexity cache invalidation ----
        self._terms_version = getattr(self, "_terms_version", 0) + 1
        self._convexity_cache_version = -1
        self._convexity_cache = None



    def get_term_by_id(self, id: int):
        """根据编号获取(项, 位置)"""
        return self.id_to_item.get(id, None)

    def get_id_by_term_and_location(self, term, location: str):
        """通过(项, 位置)找到编号"""
        return self.item_to_id.get((sympy.srepr(term), location), None)

    def display_mappings(self):
        print("Item to ID mapping:")
        for (item, loc), id in self.item_to_id.items():
            print(f"{item} at {loc} -> {id}")
        print("\nID to Item mapping:")
        for id, (item, loc) in self.id_to_item.items():
            print(f"{id} -> {item} at {loc}")
        # print(f"\nTotal items mapped: {self.counter}")
        return 1
            
    def get_item_number(self):
        return self.counter
    
    def get_from_constraints(self):
        from_constraints = []
        for id, (item, loc) in self.id_to_item.items():
            if "Constraint" in loc:
                from_constraints.append(loc.split("_")[1])
        return from_constraints
    
    def get_all_items(self):
        items = []
        for id, (item, loc) in self.id_to_item.items():
            items.append(item)
        return items

    def __repr__(self):
        s = f"QCQPProblem(name={self.name}, sense={self.obj_sense}):\n"

        if self.variables:
            s += "Variables:\n"
            for v in self.variables.values():
                s += f"  {v}\n"

        if self.matrix_variables:
            s += "MatrixVariables:\n"
            for v in self.matrix_variables.values():
                s += f"  {v}\n"

        if self.obj_expr is not None:
            s += f"Objective: {self.obj_expr}\n"

        if self.constraints:
            s += "Constraints:\n"
            for c in self.constraints:
                s += f"  {c}\n"

        if self.psd_constraints:
            s += "PSD Constraints:\n"
            for c in self.psd_constraints:
                s += f"  {c}\n"
                
        return s

    def copy(self):
        return copy.deepcopy(self)
    
    # ------------------------------------------------------
    # 🔧 1) 判断表达式是否对连续变量仿射（线性 + 常数）
    # ------------------------------------------------------
    def _is_affine(self, expr, var_syms) -> bool:
        try:
            poly = sympy.Poly(expr, *var_syms)
            return poly.total_degree() <= 1
        except (sympy.polys.polyerrors.PolynomialError,
                sympy.polys.polyerrors.CoercionFailed):
            # 不是多项式（如包含 sin），直接认为非仿射
            return False

    # ------------------------------------------------------
    # 🔧 2) 是否含有 integer/binary 之类离散变量
    # ------------------------------------------------------
    def _contains_discrete(self, expr) -> bool:
        """
        只要表达式中出现任意离散变量（binary/integer），返回 True。
        覆盖两类：
        1) 标量变量：self.variables
        2) 向量/矩阵变量：self.matrix_variables（含其 Transpose / MatrixElement 访问）
        """
        DISCRETE = {'integer', 'binary'}

        # --- 1) 标量变量 ---
        for s in expr.free_symbols:
            vinfo = self.variables.get(s.name)
            if vinfo is not None and getattr(vinfo, 'vtype', 'continuous') in DISCRETE:
                return True

        # --- 2) 向量/矩阵变量（MatrixSymbol 及其派生形态） ---
        # 收集当前表达式中出现的“底层 MatrixSymbol”
        mat_syms = set()

        # 2.1 直接出现的 MatrixSymbol
        mat_syms |= set(expr.atoms(sympy.MatrixSymbol))

        # 2.2 出现在 Transpose(...) 里的 MatrixSymbol
        for tr in expr.atoms(sympy.Transpose):
            if isinstance(tr.arg, sympy.MatrixSymbol):
                mat_syms.add(tr.arg)

        # 2.3 以 MatrixElement(parent,i,j) 的形式出现
        for me in expr.atoms(MatrixElement):
            parent = getattr(me, "parent", None) or getattr(me, "base", None) or me.args[0]
            if isinstance(parent, sympy.MatrixSymbol):
                mat_syms.add(parent)

        # 2.4 检查这些 MatrixSymbol 是否对应离散变量
        for ms in mat_syms:
            mv = self.matrix_variables.get(ms.name)
            if mv is not None and getattr(mv, 'vtype', 'continuous') in DISCRETE:
                return True

        return False

    # ------------------------------------------------------
    # 🔧 3) Abs 节点是否形如 |仿射|
    # ------------------------------------------------------
    def _is_simple_abs(self, term, var_syms) -> bool:
        if isinstance(term, sympy.Abs):
            inner = term.args[0]
            return self._is_affine(inner, var_syms)
        return False
    
    def has_nonconvex_func(self, term, variable_symbols):
        """
        判断 term 是否含有对优化变量的非凸函数调用
        """
        nonconvex_funcs = [sympy.sin, sympy.cos, sympy.tan, sympy.exp, sympy.log]

        for f in nonconvex_funcs:
            for node in term.atoms(f):
                # 判断 f(...) 中是否有优化变量
                args = node.args
                if any(v in sympy.sympify(args) for v in variable_symbols):
                    return True
        return False
    
        # ------------------------------------------------------
    # 🔧 4) 检测是否纯二次型   expr ≡ zᵀ Q z
    # ------------------------------------------------------
    def _is_quadratic_form(self, expr, cont_syms):
        """
        若 expr == zᵀQz (z 仅含连续标量变量)，返回 (True, Q)；否则 (False, None)
        """
        if not expr.is_polynomial() or sympy.total_degree(expr) != 2:
            return False, None
        try:
            z = sorted(cont_syms, key=lambda s: s.name)
            poly = sympy.Poly(expr.expand(), *z)
            Q = sympy.zeros(len(z))
            for monom, coeff in poly.terms():
                idxs = [i for i, p in enumerate(monom) if p]
                if len(idxs) == 1:        # xi²
                    i = idxs[0];  Q[i, i] += coeff
                elif len(idxs) == 2:      # xi·xj
                    i, j = idxs
                    Q[i, j] += coeff/2;  Q[j, i] += coeff/2
                else:
                    return False, None
            return True, Q
        except Exception:
            return False, None

    # ------------------------------------------------------
    # 🔧 5) 标量项的“0-1-2-3”四级凸性判定
    # ------------------------------------------------------
    def _is_convex_scalar(self, term_scalar, cont_syms):
        """
        0) xᵀQx 且 Q ⪰ 0          → True
        1) Hessian ⪰ 0            → True
        2) 线性 / 常数            → True
        3) 其它                   → False
        """
        # 0) 纯二次型
        is_q, Q = self._is_quadratic_form(term_scalar, cont_syms)
        if is_q:
            try:
                return Q.is_symmetric() and Q.is_positive_semidefinite
            except Exception:
                pass
        # 1) Hessian
        try:
            vars_ = [v for v in cont_syms if v in term_scalar.free_symbols]
            if vars_:
                H = sympy.hessian(term_scalar, vars_)
                eigs = sympy.Matrix(H).eigenvals()
                if eigs and min(float(ev) for ev in eigs) >= -1e-8:
                    return True
        except Exception:
            pass
        # 2) 线性
        try:
            if term_scalar.is_polynomial() and sympy.total_degree(term_scalar) <= 1:
                return True
        except Exception:
            pass
        # 3) 其余 → 非凸
        return False


    def _classify_scaled_xTQx(self, term, tol=1e-8):
        """
        Classify alpha*x^T*Q*x using the same recognizer as the relaxation engine.

        Returns:
          0 for convex PSD quadratic,
          2 for the zero quadratic,
          3 for indefinite / concave quadratic,
          None if the term is not recognized as x^TQx.
        """
        try:
            from autoconvexrelax.core.relaxation import RelaxationEngine
        except Exception:
            return None

        try:
            eng = RelaxationEngine()
            ok, alpha, _x_sym, Q_expr = eng.is_scaled_xTQx(term)
            if not ok:
                return None

            alpha_f = float(sympy.N(alpha))
            Qn = np.array(sympy.Matrix(Q_expr).evalf(), dtype=float)
            Qeff = 0.5 * (alpha_f * Qn + (alpha_f * Qn).T)
            eigvals = np.linalg.eigvalsh(Qeff)
            w_min = float(np.min(eigvals))
            w_max = float(np.max(eigvals))

            if abs(w_min) <= tol and abs(w_max) <= tol:
                return 2
            if w_min >= -tol:
                return 0
            return 3
        except Exception:
            return None

    def _contains_sdr_Z(self, expr) -> bool:
        """
        SDR/SDP 松弛引入的 lifting 矩阵变量 Z_* 出现时，
        相关项通常是 Trace(A*Z) 线性项或 LMI 的线性条目；
        直接视为“凸/仿射”，跳过昂贵的 PSD 判别。
        """
        import sympy
        from sympy.matrices.expressions.matexpr import MatrixElement

        # 1) 直接出现 MatrixSymbol: Trace(A * Z)
        for ms in expr.atoms(sympy.MatrixSymbol):
            v = self.matrix_variables.get(ms.name, None)
            if isinstance(v, MatrixVariableSymbol) and ms.name.startswith("Z"):
                return True

        # 2) 若 expr 被展开成了矩阵元素（更保险）
        for me in expr.atoms(MatrixElement):
            parent = me.parent
            name = None
            if isinstance(parent, sympy.MatrixSymbol):
                name = parent.name
            elif isinstance(parent, sympy.Transpose) and isinstance(parent.arg, sympy.MatrixSymbol):
                name = parent.arg.name
            if name is None:
                continue
            v = self.matrix_variables.get(name, None)
            if isinstance(v, MatrixVariableSymbol) and name.startswith("Z"):
                return True

        return False

    # ------------------------------------------------------------
    def term_class(self, term: sympy.Expr) -> int:
        """
        返回 0/1/2/3：
        0 = 标准 QCQP 二次型（如 xᵀQx / Trace(XᵀAX)，且已知 PSD）
        1 = 可判凸但不属于 0/2（如 |仿射|、或 Hessian 判凸的≤2次多项式）
        2 = 线性 / 常数
        3 = 非凸 / 未知（包括双线性 xᵀy、Trace(yᵀ*x) 等）
        """
        # [FIX] 先把 1x1 的矩阵包装解包成标量，避免把标量 Trace(...) 当成“矩阵项”误判
        if isinstance(term, sympy.MatrixBase) and term.shape == (1, 1):
            term = term[0, 0]
        elif isinstance(term, sympy.MatrixExpr) and getattr(term, "shape", None) == (1, 1):
            term = term[0, 0]


        
        # ===== FAST PATH: skip convexity check for SDR aux-Z terms =====
        if self._contains_sdr_Z(term):
            return 2   # 2 表示 convex / affine（与你现有逻辑保持一致即可）
        # =============================================================
        
        
        from sympy.matrices.expressions.matexpr import MatrixElement

        # ===== FAST PATH: affine MatrixElement terms (e.g., c*T[i,0]) =====
        if isinstance(term, MatrixElement):
            return 2
        if isinstance(term, sympy.Mul):
            has_me = any(isinstance(a, MatrixElement) for a in term.args)
            if has_me and all((a.is_number or isinstance(a, MatrixElement)) for a in term.args):
                return 2
        # =================================================================


        # ===== FAST PATH: rational / fraction terms (variable in denominator) =====
        try:
            # SymPy 的 a/b 通常会变成 a*Pow(b, -1) 或 Pow(b, -k)
            for pw in term.atoms(sympy.Pow):
                exp = pw.exp
                if exp.is_number and exp.is_negative:
                    # 分母含变量（不是纯常数）=> 直接视为非凸/未知
                    if pw.base.free_symbols:
                        return 3
        except Exception:
            pass
        # ========================================================================



        # ---------------- 收集连续标量符号 ----------------
        cont_syms = []
        for v in self.variables.values():
            if v.vtype != "continuous":
                continue
            cont_syms.append(v.sym if getattr(v, "sym", None) is not None else sympy.Symbol(v.name, real=True))
        for v in self.matrix_variables.values():
            if isinstance(v, VectorVariableSymbol):
                for i in range(v.dim):
                    cont_syms.append(v.symbol[i, 0])
            elif isinstance(v, MatrixVariableSymbol):
                for i in range(v.rows):
                    for j in range(v.cols):
                        cont_syms.append(v.symbol[i, j])

        # 记录矩阵/向量整体符号
        mat_syms = {v.symbol for v in self.matrix_variables.values()}

        # helper: MatrixElement → Dummy，便于 Hessian
        def _scalarise(expr: sympy.Expr):
            repl, extra = {}, set()
            for me in expr.atoms(sympy.matrices.expressions.matexpr.MatrixElement):
                parent = getattr(me, "parent", None) or getattr(me, "base", None) or me.args[0]
                mname  = getattr(parent, "name", getattr(parent, "label", str(parent)))
                new_s  = sympy.Dummy(f"{mname}_{me.i}_{me.j}", real=True)
                repl[me] = new_s
                extra.add(new_s)
            return expr.xreplace(repl), list(extra)

        # ---------- STEP-0：快速否定与简单肯定 ----------
        # 0-a 含离散变量 → 非凸
        if self._contains_discrete(term):
            return 3
        # 0-b 含明显非凸函数（sin/exp/log… 且作用到了变量）→ 非凸
        if self.has_nonconvex_func(term, cont_syms):
            return 3
        # 0-c |仿射| → 视作 1（凸但非线性）
        if term.has(sympy.Abs):
            return 1 if self._is_simple_abs(term, cont_syms) else 3
        # 0-d 用与 relaxation_engine 相同的 x^TQx 识别器做一次直接判定，
        # 避免 term 级凸性判断与 solver 侧 PSD 检查不一致。
        qcls = self._classify_scaled_xTQx(term)
        qcls = self._classify_scaled_xTQx(term)
        if qcls is not None:
            return qcls

        # =================================================
        # STEP-1：标准 QCQP 二次型 / 双线性 的快速判定
        # =================================================
        from collections import Counter

        quad_class = None

        def _is_mat_like(a):
            return isinstance(a, (sympy.MatrixExpr, sympy.MatrixBase))

        def _base_var_symbol(a):
            """去掉 .T 后拿到底层 MatrixSymbol；若不是【变量矩阵】则返回 None。"""
            if isinstance(a, sympy.MatrixSymbol):
                return a if a in mat_syms else None
            if isinstance(a, sympy.Transpose) and isinstance(a.arg, sympy.MatrixSymbol):
                return a.arg if a.arg in mat_syms else None
            return None

        # 抽取“标量 × Trace(...)”
        def _pull_scalar_times_trace(t):
            if isinstance(t, sympy.Trace):
                return sympy.Integer(1), t
            if isinstance(t, sympy.Mul):
                traces = [a for a in t.args if isinstance(a, sympy.Trace)]
                scalars = [a for a in t.args if not isinstance(a, sympy.Trace)]
                # 只支持恰好一个 Trace，且其它都是标量（不能再有 MatrixExpr）
                if len(traces) == 1 and all(not _is_mat_like(a) for a in scalars):
                    alpha = sympy.Mul(*scalars) if scalars else sympy.Integer(1)
                    return alpha, traces[0]
            return None, None

        # ---------- (A) Trace(...) 或 标量×Trace(...) ----------
        alpha, tr = _pull_scalar_times_trace(term)
        if tr is not None:
            inner = tr.arg
            # 只取矩阵因子（把 4 这类标量剥掉）
            if isinstance(inner, sympy.MatMul):
                mat_args = [a for a in inner.args if _is_mat_like(a)]
            else:
                mat_args = [inner] if _is_mat_like(inner) else []

            # 统计出现的“变量矩阵”（去 .T）
            var_list = []
            for a in mat_args:
                base = _base_var_symbol(a)
                if base is not None:
                    var_list.append(base)

            # 单变量线性：Trace(C * Z) 或 Trace(Z * C) 仅出现一次 → 2
            if var_list and len(set(var_list)) == 1 and Counter(var_list)[var_list[0]] == 1:
                return 2

            # Frobenius²：Trace(x*x.T) / Trace(x.T*x) → 0
            if len(mat_args) == 2:
                A, B = mat_args
                same_pair = (
                    (isinstance(A, sympy.Transpose) and A.arg == B) or
                    (isinstance(B, sympy.Transpose) and B.arg == A)
                )
                if same_pair:
                    # alpha * Trace(x.T*x)  : convex if alpha>0, concave if alpha<0
                    if alpha.is_zero:
                        return 2
                    if alpha.is_positive:
                        return 0
                    if alpha.is_negative:
                        return 3
                    # Fallback for numeric alpha
                    try:
                        aval = float(alpha)
                        return 0 if aval > 0 else (3 if aval < 0 else 2)
                    except Exception:
                        return 3
                # 不同向量内积 Trace(y.T*x) / Trace(x*y.T) → 非凸
                is_vec_ip = (
                    (isinstance(A, sympy.Transpose) and isinstance(A.arg, sympy.MatrixSymbol)
                    and isinstance(B, sympy.MatrixSymbol)) or
                    (isinstance(B, sympy.Transpose) and isinstance(B.arg, sympy.MatrixSymbol)
                    and isinstance(A, sympy.MatrixSymbol))
                )
                if is_vec_ip and (not same_pair):
                    return 3

            # x^T Q x：Trace(x.T*Q*x)（三因子，Q 对称 PSD）→ 0；NSD → 3
            elif len(mat_args) == 3:
                L, Q, R = mat_args
                transpose_pair = (
                    (isinstance(L, sympy.Transpose) and L.arg == R) or
                    (isinstance(R, sympy.Transpose) and R.arg == L)
                )
                if transpose_pair and getattr(Q, "is_symmetric", False):
                    # 先处理 α = 0 的情况：整个项就是常数 0，算线性/常数
                    if alpha.is_zero:
                        return 2

                    # 把 α 吸进 Q，检查 αQ 的半正定性
                    Q_eff = alpha * Q

                    psd = nsd = None
                    try:
                        psd = Q_eff.is_positive_semidefinite
                    except Exception:
                        psd = None

                    try:
                        nsd = Q_eff.is_negative_semidefinite
                    except Exception:
                        nsd = None

                    # ① 先走符号判定（如果没炸）
                    if psd is True:
                        return 0
                    if nsd is True:
                        return 3

                    # ② 再走你已经写了的 numeric fallback（αBB 的 Qpsd 大概率会在这里正确被判 PSD）
                    num_psd = self._is_psd_numeric(Q_eff, tol=1e-8)
                    if num_psd is True:
                        return 0
                    if num_psd is False:
                        num_nsd = self._is_psd_numeric(-Q_eff, tol=1e-8)
                        if num_nsd is True:
                            return 3


                    if psd is True:
                        # αQ ⪰ 0 → 整体是凸二次型
                        return 0
                    elif nsd is True:
                        # αQ ⪯ 0 → 整体是凹二次型，当成“非凸/有害”
                        return 3
                    # α 符号不确定 或 Q_eff 性质不确定，交给后面的通用逻辑/Hessian 兜底
                # y≠x 的 y^T*A*x → 非凸
                if transpose_pair:
                    same_lr = (
                        (isinstance(L, sympy.Transpose) and (L.arg == R)) or
                        (isinstance(R, sympy.Transpose) and (R.arg == L))
                    )
                    if not same_lr:
                        return 3

            # 兜底：同一“变量矩阵”出现 ≥2 次但未命中上面模式 → 多半双线性 → 3
            if var_list and any(Counter(var_list)[v] >= 2 for v in set(var_list)):
                return 3


        # ---------- (B) 其它：MatMul 三因子 y^T*Q*x / x^T*Q*x ----------
        elif isinstance(term, sympy.MatMul):
            # 只看矩阵因子
            mats = [a for a in term.args if _is_mat_like(a)]
            
            # === [新增] 两因子：常数矩阵/向量 × 决策向量/矩阵 → 线性(2) ===
            if len(mats) == 2:
                A, B = mats

                def _base(a):
                    return a.arg if isinstance(a, sympy.Transpose) else a

                A0, B0 = _base(A), _base(B)

                # 是否是“你的决策矩阵/向量变量”
                is_varA = isinstance(A0, sympy.MatrixSymbol) and (A0 in mat_syms)
                is_varB = isinstance(B0, sympy.MatrixSymbol) and (B0 in mat_syms)

                # 是否是“常数矩阵/向量”（稠密 Matrix，且无自由符号）
                is_constA = isinstance(A0, sympy.MatrixBase) and (not A0.free_symbols)
                is_constB = isinstance(B0, sympy.MatrixBase) and (not B0.free_symbols)

                # 形如  b.T * x   或   x.T * b   或   C * X   /   X * C
                if (is_varA and is_constB) or (is_varB and is_constA):
                    # 若是 1×1 标量（常见于向量内积形式），直接线性判定
                    try:
                        if getattr(term, "shape", None) == (1, 1):
                            return 2
                    except Exception:
                        # 兜底：形状不明，但本质仍是线性
                        return 2
                    
            if len(mats) == 3:
                L, Q, R = mats

                # 形如 x·Q·x（无转置隔离）→ 直接判非凸（我们只接受 x.T*Q*x）
                if L == R:
                    return 3  

                # 是否是 yᵀ * Q * x / xᵀ * Q * y 这种三因子结构
                transpose_pair = (
                    isinstance(L, sympy.Transpose) and isinstance(L.arg, sympy.MatrixSymbol)
                    and isinstance(R, sympy.MatrixSymbol)
                ) or (
                    isinstance(R, sympy.Transpose) and isinstance(R.arg, sympy.MatrixSymbol)
                    and isinstance(L, sympy.MatrixSymbol)
                )

                if transpose_pair:
                    # 要求左右是同一个向量（xᵀ Q x），否则视作双线性 yᵀ Q x
                    same = (isinstance(L, sympy.Transpose) and (L.arg == R)) or \
                           (isinstance(R, sympy.Transpose) and (R.arg == L))
                    if not same:
                        return 3  # yᵀ*Q*x，双线性 → 非凸

                    # 到这里：就是 xᵀ * Q * x 这一类二次型
                    # -----------------------------------------
                    # ① 先用 SymPy 自己的符号信息（适合符号 Q）
                    # -----------------------------------------
                    psd = getattr(Q, "is_positive_semidefinite", None)
                    nsd = getattr(Q, "is_negative_semidefinite", None)
                    if psd is True:
                        return 0      # xᵀQx, Q ⪰ 0 → 标准凸二次型
                    if nsd is True:
                        return 3      # xᵀQx, Q ⪯ 0 → 凹的，当成“非凸/有害”

                    # -----------------------------------------
                    # ② 不管 Q 是否被标记为 symmetric，
                    #    只要 Q 是纯常数矩阵，就数值检查特征值
                    # -----------------------------------------
                    num_psd = self._is_psd_numeric(Q, tol=1e-8)
                    if num_psd is True:
                        return 0      # 所有特征值 >= -1e-8 → 当作 PSD
                    if num_psd is False:
                        # 如果 Q 不是 PSD，但 -Q 是 PSD，则相当于凹二次型
                        num_nsd = self._is_psd_numeric(-Q, tol=1e-8)
                        if num_nsd is True:
                            return 3  # 凹二次型，当成非凸

                    # ③ 到这一步：Q 里可能还有符号或者结构更复杂，
                    #    留给后面的 Hessian / 线性 / 默认分支处理

        # =================================================
        # STEP-2：Hessian（≤2 次多项式）判凸 → 1
        # =================================================
        # 取用于 Hessian 的标量表达式
        if isinstance(term, sympy.MatrixExpr):
            if term.shape == (1, 1):
                term_scalar = term[0, 0]
            elif isinstance(term, sympy.Trace):
                term_scalar = term
            else:
                term_scalar = term
        else:
            term_scalar = term

        term_s, extra_syms = _scalarise(term_scalar)
        free_syms = set(term_s.free_symbols)
        free_by_name = {
            getattr(s, "name", None): s
            for s in free_syms
            if isinstance(s, sympy.Symbol) and getattr(s, "name", None) is not None
        }
        vars_in = []
        seen_vars = set()
        for s in cont_syms + extra_syms:
            matched = s if s in free_syms else free_by_name.get(getattr(s, "name", None))
            if matched is None or matched in seen_vars:
                continue
            vars_in.append(matched)
            seen_vars.add(matched)

        # 出现同一矩阵符号多次 → 双线性 → 非凸
        mat_in_term = [sym for sym in mat_syms if sym in term.free_symbols]
        if len(mat_in_term) >= 1:
            sym_list = list(term.free_symbols)
            if any(Counter(sym_list)[m] > 1 for m in mat_in_term):
                return 3

        # ---------- FAST PATH: 线性先判（避免对线性大表达式构 Hessian） ----------
        try:
            if vars_in and term_s.is_polynomial(*vars_in):
                deg = sympy.total_degree(term_s, *vars_in)
                if deg <= 1:
                    return 2
        except Exception:
            pass

        # ---------- Hessian PSD test：只对“小规模二次”做 ----------
        try:
            if vars_in and term_s.is_polynomial(*vars_in) and sympy.total_degree(term_s, *vars_in) == 2:
                # 阈值：变量太多直接跳过 Hessian（避免 11s 爆炸）
                MAX_HESS_VARS = 12
                if len(vars_in) <= MAX_HESS_VARS:
                    H = sympy.hessian(term_s, vars_in)

                    # 只在 Hessian 为常数矩阵时做数值特征值，避免 symbolic eigenvals
                    Hm = sympy.Matrix(H)
                    if not Hm.free_symbols:
                        import numpy as np
                        Hn = np.array(Hm.tolist(), dtype=float)
                        Hn = 0.5 * (Hn + Hn.T)  # 数值对称化
                        if np.min(np.linalg.eigvalsh(Hn)) >= -1e-10:
                            return 1
        except Exception:
            pass


        # =================================================
        # STEP-3：线性 / 常数 → 2
        # =================================================
        try:
            if not vars_in:
                # 若含整体矩阵变量（而标量变量集为空），保守：3
                has_matrix_var = any(term.has(ms) for ms in mat_syms)
                return 3 if has_matrix_var else 2
            if term_s.is_polynomial(*vars_in) and sympy.total_degree(term_s, *vars_in) <= 1:
                return 2
        except Exception:
            pass

        # =================================================
        # STEP-4：默认 非凸 / 未知 → 3
        # =================================================
        return 3
    
    # def get_convexity_classes(self) -> List[int]:
    #     v = getattr(self, "_terms_version", 0)
    #     if getattr(self, "_convexity_cache", None) is not None and getattr(self, "_convexity_cache_version", -1) == v:
    #         return self._convexity_cache

    #     classes = []
    #     for idx in sorted(self.id_to_item.keys()):
    #         term, _ = self.id_to_item[idx]
    #         classes.append(self.term_class(term))

    #     self._convexity_cache = classes
    #     self._convexity_cache_version = v
    #     return classes

    def get_convexity_classes(self) -> List[int]:
        import os, time

        v = getattr(self, "_terms_version", 0)
        if getattr(self, "_convexity_cache", None) is not None and getattr(self, "_convexity_cache_version", -1) == v:
            return self._convexity_cache

        # === profiling 开关：不想打印就把环境变量关掉 ===
        PROFILE = os.getenv("CONVEXITY_PROFILE", "1") == "1"
        THRESH  = float(os.getenv("CONVEXITY_PROFILE_THRESH", "0.05"))  # 秒；只打印超过阈值的
        TOPK    = int(os.getenv("CONVEXITY_PROFILE_TOPK", "10"))

        slow = []  # (dt, idx, cls, term_str)

        def _short_term_str(t):
            s = str(t)
            s = " ".join(s.split())  # 压缩空白
            if len(s) > 300:
                s = s[:300] + " ... <truncated>"
            return s

        classes = []
        for idx in sorted(self.id_to_item.keys()):
            term, _ = self.id_to_item[idx]

            if PROFILE:
                t0 = time.perf_counter()

            cls = self.term_class(term)
            classes.append(cls)

            if PROFILE:
                dt = time.perf_counter() - t0
                if dt >= THRESH:
                    ts = _short_term_str(term)
                    slow.append((dt, idx, cls, ts))
                    print(f"[convexity_profile] dt={dt:.6f}s idx={idx} class={cls} term={ts}")

        if PROFILE and slow:
            slow.sort(key=lambda x: x[0], reverse=True)
            print("\n[convexity_profile] ==== TOP slow terms ====")
            for dt, idx, cls, ts in slow[:TOPK]:
                print(f"  dt={dt:.6f}s idx={idx} class={cls} term={ts}")
            print("[convexity_profile] ========================\n")

        self._convexity_cache = classes
        self._convexity_cache_version = v
        return classes

    
    def get_convexity_flags(self) -> list:
        return [cls in (0, 1, 2) for cls in self.get_convexity_classes()]

    
    def is_convex(self) -> bool:
        flags = self.get_convexity_flags()
        # 输出每个项及其对应是否为凸
        # for (id, (term, loc)), flag in zip(sorted(self.id_to_item.items()), flags):
        #     print(f"[ID {id}] {loc:20} | {str(term):20} → {'凸' if flag else '非凸'}")
        return all(flags)
    
    def get_term_convexity(self) -> None:
        classes = self.get_convexity_classes()
        label = {
            0: "0(标准二次型/PSD)",
            1: "1(可判凸: |仿射|/Hessian凸)",
            2: "2(线性/常数)",
            3: "3(非凸/未知)"
        }
        for (id_, (term, loc)), cls in zip(sorted(self.id_to_item.items()), classes):
            print(f"[ID {id_}] {loc:20} | {str(term):30} → {label[cls]}")
    
    # def get_term_convexity(self) -> bool:
    #     flags = self.get_convexity_flags()
    #     # 输出每个项及其对应是否为凸
    #     for (id, (term, loc)), flag in zip(sorted(self.id_to_item.items()), flags):
    #         print(f"[ID {id}] {loc:20} | {str(term):20} → {'凸' if flag else '非凸'}")
    
    def __eq__(self, other):
        if not isinstance(other, QCQPProblem):
            return False
        return (
            self.obj_sense == other.obj_sense and
            str(self.obj_expr) == str(other.obj_expr) and
            list(self.variables.keys()) == list(other.variables.keys()) and
            list(self.matrix_variables.keys()) == list(other.matrix_variables.keys()) and
            all(str(self.variables[k]) == str(other.variables[k]) for k in self.variables) and
            all(str(self.matrix_variables[k]) == str(other.matrix_variables[k]) for k in self.matrix_variables) and
            [str(c) for c in self.constraints] == [str(c) for c in other.constraints] and
            [str(p) for p in self.psd_constraints] == [str(p) for p in other.psd_constraints]
        )
        
    def __hash__(self):
        return hash(str(self))
    
    @property
    def problem_type(self) -> str:
        """
        如果出现 add_vector_variable / add_matrix_variable 创建的变量，
        就判为 'vector'；否则 'scalar'
        """
        return 'vector' if self.matrix_variables else 'scalar'
    
    
if __name__ == "__main__":
    
    def _vec(prob: QCQPProblem, name: str, dim: int, lb=-2, ub=2, vtype=None):
        """一次性创建 dim×1 向量决策变量；返回 MatrixSymbol。"""
        return prob.add_vector_variable(name, dim, lb=lb, ub=ub, vtype=vtype)

    # P2  —— 非正定Q，让 x 变成 **integer**
    p = QCQPProblem("G3_P2")
    n = 3
    x = _vec(p, "x", n, lb=-2, ub=2, vtype="integer")
    y = _vec(p, "y", n)
    Qc = sympy.Matrix([[2, 1, 0],
                    [1, -3, 1],
                    [0, 1, -1]])
    obj = 0.5 * sympy.Trace(x.T * Qc * x) - sympy.Trace(y.T * Qc * x)
    p.set_objective(obj, "min")
    p.add_constraint(y.T * y, "<=", 4)
    p.map_all_terms()

    p.display_mappings()
