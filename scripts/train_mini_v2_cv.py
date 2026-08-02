#!/usr/bin/env python3
"""Explicit fold launcher; it never bypasses V2 preflight or overfit gates."""
from __future__ import annotations
import argparse, subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--", dest="rest", nargs=argparse.REMAINDER)
    args, unknown = parser.parse_known_args()
    extra = args.rest or unknown
    if "--fold" in extra:
        extra = extra[extra.index("--fold") + 2:]
    command = ["python", "-m", "scripts.train_mini_v2", "--fold", str(args.fold)] + extra
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__": main()
