#!/usr/bin/env python3
"""Fail-closed V2 inference entry point."""
from __future__ import annotations

import argparse
import json

from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.evaluation_mini_v2 import validate_evaluation_report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--sequence", required=True); parser.add_argument("--k", type=int, required=True); parser.add_argument("--p", type=int, required=True); parser.add_argument("--mode", choices=("unassisted", "assisted"), default="unassisted"); args = parser.parse_args()
    CandidateCondition(args.sequence, args.k, args.p)
    raise RuntimeError("V2 rollout requires a trained, provenance-valid checkpoint and strict evaluation target set")


if __name__ == "__main__":
    main()
