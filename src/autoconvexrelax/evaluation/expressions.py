# expression_utils.py
import sympy as sp
from sympy import Trace, MatMul, MatAdd, Transpose, MatrixExpr, MatrixBase, Add
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.matrices.expressions import MatrixSymbol

def normalize_expr(expr):
    out = expr
    # Trace -> scalar
    if isinstance(expr, Trace):
        A = expr.args[0]

        # 1) Trace(Z_x) where Z_x is MatrixSymbol / MatrixVariableSymbol
        #    这种形式无需再“规范化”，直接返回即可
        if isinstance(A, (MatrixSymbol,)) or getattr(A, "is_Matrix", False):
            return sp.Trace(A)

        # 1.5) Trace(custom matrix-like): 只要有 shape=(n,n) 就放行
        #      例如 MatrixVariableSymbol / 其它自定义矩阵符号
        if hasattr(A, "shape") and isinstance(getattr(A, "shape"), tuple) and len(getattr(A, "shape")) == 2:
            return sp.Trace(A)

        # 2) Trace(MatrixExpr): 只递归 normalize 里面的表达式，不强制展开
        if isinstance(A, MatrixExpr):
            return sp.Trace(normalize_expr(A))

        # 3) 兜底：如果 trace 里面是纯标量（极少），也允许
        if A.is_scalar:
            return sp.Trace(A)

        raise NotImplementedError(f"Un-normalizable Trace: {expr}")

    # 1x1 MatrixExpr -> scalar
    if isinstance(expr, MatrixExpr):
        if _is_1x1_matrixexpr(expr):
            return normalize_expr(expr[0, 0])
        raise NotImplementedError(f"Non-scalar MatrixExpr: {expr}")

    # recurse
    if isinstance(expr, sp.Add):
        return sp.Add(*[normalize_expr(a) for a in expr.args])
    if isinstance(expr, sp.Mul):
        return sp.Mul(*[normalize_expr(a) for a in expr.args])


    try:
        # 如果 out 仍是 MatrixExpr 且非 1x1，这里会走 except
        if not isinstance(out, MatrixExpr):
            out = sp.expand(out)
    except Exception:
        pass
    return out

def _is_1x1_matrixexpr(e) -> bool:
    shp = getattr(e, "shape", None)
    return shp == (1, 1)

def _vec_elem(v, i):
    """Return MatrixElement(v, i, 0) for column vector v (MatrixSymbol)."""
    return MatrixElement(v, i, 0)

def _as_const_matrix(M):
    """Try convert to dense MatrixBase if possible."""
    if isinstance(M, MatrixBase):
        return M
    try:
        # some MatrixExpr may be convertible
        MM = sp.Matrix(M)
        if isinstance(MM, MatrixBase):
            return MM
    except Exception:
        pass
    return None

def _matmul_1x1_to_scalar_expr(mm):
    """Convert a 1x1 MatMul (or MatrixExpr) to scalar Sympy expression, for common quadratic/bilinear patterns."""
    if isinstance(mm, Trace):
        mm = mm.arg
    if isinstance(mm, MatrixBase) and getattr(mm, "shape", None) == (1, 1):
        return mm[0, 0]
    if not isinstance(mm, MatMul):
        # fallback: if it's a 1x1 MatrixExpr, try take [0,0]
        if isinstance(mm, MatrixExpr) and _is_1x1_matrixexpr(mm):
            try:
                return mm[0, 0]
            except Exception:
                pass
        return None

    mats = [a for a in mm.args if isinstance(a, (MatrixExpr, MatrixBase, Transpose))]
    if len(mats) == 2 and isinstance(mats[0], Transpose):
        # v.T * w  (including w==v)
        v = mats[0].arg
        w = mats[1]
        if getattr(v, "shape", None) and getattr(w, "shape", None):
            n = v.shape[0] if v.shape[1] == 1 else v.shape[1]
            # assume both are vectors and compatible
            return sp.Add(*[_vec_elem(v, i) * _vec_elem(w, i) for i in range(n)])

    if len(mats) == 3 and isinstance(mats[0], Transpose):
        # v.T * Q * w   where Q is constant matrix
        v = mats[0].arg
        Q = _as_const_matrix(mats[1])
        w = mats[2]
        if Q is None:
            return None
        n = v.shape[0] if v.shape[1] == 1 else v.shape[1]
        m = w.shape[0] if w.shape[1] == 1 else w.shape[1]
        if Q.shape != (n, m):
            return None
        terms = []
        for i in range(n):
            for j in range(m):
                qij = Q[i, j]
                try:
                    qij = float(sp.N(qij))
                except Exception:
                    # symbolic constant not supported
                    return None
                if abs(qij) < 1e-15:
                    continue
                terms.append(sp.Float(qij) * _vec_elem(v, i) * _vec_elem(w, j))
        return sp.Add(*terms) if terms else sp.Float(0.0)

    return None

def _trace_to_scalar_expr(tr: Trace):
    inner = tr.arg

    # Case A: Trace(Z) where Z is a matrix symbol (or custom matrix-like with shape)
    # Expand to sum of diagonal entries: sum_i Z[i,i]
    if (isinstance(inner, (MatrixSymbol, MatrixExpr)) or hasattr(inner, "shape")):
        shape = getattr(inner, "shape", None)
        if isinstance(shape, tuple) and len(shape) == 2 and shape[0] == shape[1]:
            try:
                n = int(shape[0])
                return Add(*[inner[i, i] for i in range(n)])
            except Exception:
                pass

    # trace(A + B) = trace(A) + trace(B)
    if isinstance(inner, MatAdd):
        parts = [_trace_to_scalar_expr(Trace(a)) for a in inner.args]
        return sp.Add(*parts)

    # === Case 1: Trace(v.T * v) ===
    if isinstance(inner, MatMul):
        mats = inner.args
        if len(mats) == 2 and isinstance(mats[0], Transpose):
            v = mats[0].arg
            w = mats[1]
            if v == w and isinstance(v, MatrixSymbol):
                n = v.shape[0]
                return sp.Add(*[MatrixElement(v, i, 0)**2 for i in range(n)])

    # === Case 2: Trace(C * Z) or Trace(Z * C), C constant, Z matrix variable ===
    if isinstance(inner, MatMul):
        mats = inner.args

        # Trace(C * Z)
        if len(mats) == 2:
            C = _as_const_matrix(mats[0])
            Z = mats[1]
            if C is not None and isinstance(Z, MatrixExpr):
                rows, cols = Z.shape
                terms = []
                for i in range(rows):
                    for j in range(cols):
                        cij = C[i, j]
                        try:
                            cij = float(sp.N(cij))
                        except Exception:
                            return None
                        if abs(cij) < 1e-15:
                            continue
                        terms.append(sp.Float(cij) * MatrixElement(Z, j, i))
                return sp.Add(*terms) if terms else sp.Float(0.0)

        # Trace(Z * C)
        if len(mats) == 2:
            Z = mats[0]
            C = _as_const_matrix(mats[1])
            if C is not None and isinstance(Z, MatrixExpr):
                rows, cols = Z.shape
                terms = []
                for i in range(rows):
                    for j in range(cols):
                        cij = C[j, i]
                        try:
                            cij = float(sp.N(cij))
                        except Exception:
                            return None
                        if abs(cij) < 1e-15:
                            continue
                        terms.append(sp.Float(cij) * MatrixElement(Z, i, j))
                return sp.Add(*terms) if terms else sp.Float(0.0)

    # === Case 3: Trace(1x1 Matrix) ===
    if isinstance(inner, MatrixBase) and getattr(inner, "shape", None) == (1, 1):
        return inner[0, 0]

    # === Fallback: try generic 1x1 MatMul ===
    s = _matmul_1x1_to_scalar_expr(inner)
    if s is not None:
        return s

    # last resort
    if isinstance(inner, MatrixExpr) and _is_1x1_matrixexpr(inner):
        try:
            return inner[0, 0]
        except Exception:
            pass

    return None

def _dense_from_matsym(A: sp.MatrixSymbol) -> sp.Matrix:
            r, c = map(int, A.shape)
            return sp.Matrix([[MatrixElement(A, i, j) for j in range(c)] for i in range(r)])
        
def _split_scalar_matmul(mm: sp.MatMul):
            scalar = sp.Integer(1)
            mats = []
            for a in mm.args:
                if a.is_Number:
                    scalar *= a
                else:
                    mats.append(a)
            return scalar, mats
        
def _try_schur_block(expr):
            # Match: Z - x*x.T  (or Z + (-1)*x*x.T), where Z is (n,n), x is (n,1)
            Z_sym = None
            x_sym = None
            coeff_Z = sp.Integer(0)
            coeff_xxt = sp.Integer(0)

            terms = expr.args if isinstance(expr, sp.MatAdd) else (expr,)
            for t in terms:
                if isinstance(t, sp.MatrixSymbol):
                    Z_sym = t
                    coeff_Z += 1
                elif isinstance(t, sp.MatMul):
                    s, mats = _split_scalar_matmul(t)
                    # scalar * Z
                    if len(mats) == 1 and isinstance(mats[0], sp.MatrixSymbol):
                        Z_sym = mats[0]
                        coeff_Z += s
                    # scalar * x * x.T
                    elif len(mats) == 2 and isinstance(mats[0], sp.MatrixSymbol) and isinstance(mats[1], sp.Transpose) and mats[1].args[0] == mats[0]:
                        x_sym = mats[0]
                        coeff_xxt += s

            if Z_sym is None or x_sym is None:
                return None
            if coeff_Z != 1 or coeff_xxt != -1:
                return None

            n1, n2 = map(int, Z_sym.shape)
            nx, mx = map(int, x_sym.shape)
            if n1 != n2 or mx != 1 or nx != n1:
                return None

            Z_dense = _dense_from_matsym(Z_sym)  # n x n
            x_col = sp.Matrix([MatrixElement(x_sym, i, 0) for i in range(nx)])  # n x 1

            top = sp.Matrix.hstack(sp.Matrix([[sp.Integer(1)]]), x_col.T)      # 1 x (n+1)
            bottom = sp.Matrix.hstack(x_col, Z_dense)                          # n x (n+1)
            return sp.Matrix.vstack(top, bottom)                               # (n+1) x (n+1)
        
def _to_affine_psd_matrix(mat_expr):
            # 1) Explicit Matrix
            if isinstance(mat_expr, sp.MatrixBase):
                return sp.Matrix(mat_expr)
            # 2) Matrix symbol: treat as itself
            if isinstance(mat_expr, sp.MatrixSymbol):
                return _dense_from_matsym(mat_expr)
            # 3) Try Schur block for Z - x*x.T
            blk = _try_schur_block(mat_expr)
            if blk is not None:
                return blk
            # 4) Last resort: try to coerce to Matrix (may fail for non-affine matrix expressions)
            try:
                return sp.Matrix(mat_expr)
            except Exception:
                return None