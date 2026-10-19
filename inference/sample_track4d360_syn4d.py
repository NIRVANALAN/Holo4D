"""Compatibility entrypoint; implementation lives in the canonical driver."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples/wan2.1_fun"))
from eval_track4d360_hybrid_dense_syn4d import *  # noqa: F401,F403,E402

if __name__ == "__main__":
    main()
