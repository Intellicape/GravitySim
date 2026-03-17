import argparse
import app.model as _model
from app.ui_pygame import NBodyUI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-Body gravitational simulator")
    parser.add_argument("--scene", default="", metavar="NAME",
                        help="Scene name to load from scenes/ (e.g. two_body)")
    parser.add_argument("--name", default="current", metavar="NAME",
                        help="Save-file name under scenes/ (default: current)")
    parser.add_argument("--numba", action="store_true",
                        help="Enable Numba JIT acceleration (requires numba installed)")
    args = parser.parse_args()

    if args.numba:
        ok = _model.enable_numba()
        print(f"[numba] {'включена' if ok else 'не удалось включить, используется NumPy'}")

    NBodyUI().run(scene=args.scene, save_name=args.name)
