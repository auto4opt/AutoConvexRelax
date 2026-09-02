import os
import pickle
from pathlib import Path

from autoconvexrelax.evaluation.real_applications.instances import build_real_application_instances
from autoconvexrelax.paths import OUTPUT_ROOT, PROJECT_ROOT


def build_real_applications_dataset(
    out_path: str = str(OUTPUT_ROOT / "data" / "real_applications.pkl"),
):
    problems = build_real_application_instances()

    groups = [problems]
    out_file = Path(out_path)
    if not out_file.is_absolute():
        out_file = PROJECT_ROOT / out_file

    os.makedirs(out_file.parent, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(groups, f)

    print(f"[OK] Saved {len(problems)} problems to {out_file}")
    return str(out_file)


if __name__ == "__main__":
    build_real_applications_dataset()
