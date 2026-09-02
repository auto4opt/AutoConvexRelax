"""AutoConvexRelax: learned construction of convex relaxations."""

import sys
import types

from autoconvexrelax.core import problem as _problem_module
from autoconvexrelax.core.problem import QCQPProblem
from autoconvexrelax.core.relaxation import RelaxationEngine


# Existing datasets were serialized before the src-layout cleanup. Register the
# former module path so those QCQPProblem pickles remain readable.
_legacy_qcqp = sys.modules.setdefault("QCQP", types.ModuleType("QCQP"))
_legacy_qcqp.__path__ = []
sys.modules.setdefault("QCQP.problem_structure", _problem_module)
sys.modules.setdefault("problem_structure", _problem_module)

__all__ = ["QCQPProblem", "RelaxationEngine"]
