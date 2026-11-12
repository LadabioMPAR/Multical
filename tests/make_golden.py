"""
Regenerate tests/golden/reference.npz from the current code.

Run from the repository root, and only when a change in the numbers is intended:

    python -m tests.make_golden --force
"""
import os
import sys

import numpy as np

from ._cases import compute_all

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "reference.npz")


def main(argv):
    if os.path.exists(GOLDEN_PATH) and "--force" not in argv:
        print(f"{GOLDEN_PATH} already exists. Use --force to overwrite it.")
        return 1
    results = compute_all()
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    np.savez_compressed(GOLDEN_PATH, **results)
    print(f"Saved {len(results)} arrays to {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
