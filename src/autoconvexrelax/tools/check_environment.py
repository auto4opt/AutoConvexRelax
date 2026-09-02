#!/usr/bin/env python3
"""Check software and external artifacts required by AutoConvexRelax."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


PACKAGES = {
    "numpy": "NumPy",
    "sympy": "SymPy",
    "torch": "PyTorch",
    "torch_geometric": "PyTorch Geometric",
    "matplotlib": "Matplotlib",
    "pandas": "pandas",
    "networkx": "NetworkX",
    "gurobipy": "Gurobi Python interface",
    "mosek": "MOSEK Python interface",
    "pyscipopt": "PySCIPOpt",
}


def package_status(module_name: str, display_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"[MISSING] {display_name}: {exc.__class__.__name__}")
        return False
    version = getattr(module, "__version__", "installed")
    print(f"[OK]      {display_name}: {version}")
    return True


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    print(f"Python: {platform.python_version()}")
    print(f"Repository: {repo_root}")

    all_present = True
    for module_name, display_name in PACKAGES.items():
        all_present &= package_status(module_name, display_name)

    source_root = str(repo_root / "src")
    if source_root in sys.path:
        print("[OK]      src/ package path is configured")
    else:
        print('[MISSING] source path; use the commands under run/ or export PYTHONPATH="$PWD/src"')
        all_present = False

    license_path = os.environ.get("MOSEKLM_LICENSE_FILE")
    if license_path and Path(license_path).expanduser().is_file():
        print("[OK]      MOSEKLM_LICENSE_FILE points to an external file")
    else:
        print("[MISSING] MOSEKLM_LICENSE_FILE is unset or does not point to a file")
        all_present = False

    print("Environment check passed." if all_present else "Environment check found missing requirements.")
    return 0 if all_present else 1


if __name__ == "__main__":
    sys.exit(main())
