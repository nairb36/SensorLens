#!/usr/bin/env python3
"""
Patch motmetrics for numpy >= 1.24 compatibility.

Fixes: np.bool (removed in numpy 1.24) -> np.bool_
Run once after pip install. Idempotent — safe to re-run.
"""

import importlib
import os


def main():
    mod = importlib.import_module("motmetrics")
    mot_py = os.path.join(os.path.dirname(mod.__file__), "mot.py")

    with open(mot_py) as f:
        content = f.read()

    old = "dtype=np.bool)"
    new = "dtype=np.bool_)"

    if old in content and new not in content:
        content = content.replace(old, new)
        with open(mot_py, "w") as f:
            f.write(content)
        print(f"Patched {mot_py}")
    else:
        print(f"Already patched (or not needed): {mot_py}")


if __name__ == "__main__":
    main()
