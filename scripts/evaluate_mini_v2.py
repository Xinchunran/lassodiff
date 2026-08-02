#!/usr/bin/env python3
"""Validate a paired mini_dev report; strict results are mandatory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lassodiff.evaluation_mini_v2 import validate_evaluation_report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--report", required=True); args = parser.parse_args()
    report = json.loads(Path(args.report).read_text()); validate_evaluation_report(report); print(json.dumps({"status": "PASS", "strict_required": True}, sort_keys=True))


if __name__ == "__main__":
    main()
