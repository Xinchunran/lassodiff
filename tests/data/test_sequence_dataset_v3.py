from __future__ import annotations

import pytest

from lassodiff.data.sequence_v3 import SequenceExample, validate_sequence_examples


def test_weak_lassopred_candidate_cannot_be_the_only_positive_evidence():
    rows = [SequenceExample("x", "ACDEFG", 1, "positive", "weak_lassopred_candidate", "x", 2, 5)]
    with pytest.raises(RuntimeError, match="verified positive evidence"):
        validate_sequence_examples(rows)


def test_hard_mutant_must_share_group_with_source_positive():
    rows = [
        SequenceExample("p", "ACDEFG", 1, "positive", "verified_lasso", "family-a", 2, 5),
        SequenceExample("m", "ACNQFG", 0, "hard_mutant", "derived_mutant", "different"),
    ]
    with pytest.raises(RuntimeError, match="source group"):
        validate_sequence_examples(rows, source_groups={"m": "family-a"})
