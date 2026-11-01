"""Convenience entry point for the public DUTS archive mirror."""

import argparse
from download_data import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data")
    args = parser.parse_args()
    prepare(args.data_root, mirror=True)
