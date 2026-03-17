import argparse
from app.ui_pygame import NBodyUI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-Body gravitational simulator")
    parser.add_argument("--scene", default="", metavar="NAME",
                        help="Scene name to load from scenes/ (e.g. two_body)")
    parser.add_argument("--name", default="current", metavar="NAME",
                        help="Save-file name under scenes/ (default: current)")
    args = parser.parse_args()

    NBodyUI().run(scene=args.scene, save_name=args.name)
