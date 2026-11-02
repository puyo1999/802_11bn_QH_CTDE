"""STA displacement used by experiments/staleness_eval.py."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from staleness_eval import displace, misassociated_share  # noqa: E402
from src.envs.topology_registry import get_registry  # noqa: E402


def test_displace_moves_every_sta_by_shift_and_keeps_association():
    base = get_registry().get("T6")
    moved = displace(base, 3.0, np.random.default_rng(0))
    for old_row, new_row in zip(base.sta_positions, moved.sta_positions):
        assert len(old_row) == len(new_row)
        for old, new in zip(old_row, new_row):
            assert abs(np.linalg.norm(np.asarray(new) - np.asarray(old)) - 3.0) < 1e-9
    assert moved.sta_overlap == base.sta_overlap
    np.testing.assert_allclose(moved.ap_positions, base.ap_positions)
    # The base topology is untouched.
    assert get_registry().get("T6").sta_positions[0][0] is not moved.sta_positions[0][0]


def test_zero_shift_is_identity_and_misassociation_grows_with_shift():
    base = get_registry().get("T3")
    np.testing.assert_allclose(np.concatenate(displace(base, 0.0, np.random.default_rng(1)).sta_positions),
                               np.concatenate(base.sta_positions))
    far = np.mean([misassociated_share(displace(base, 25.0, np.random.default_rng(s))) for s in range(10)])
    assert far > misassociated_share(base)
