"""Tests for the jump-slot capture processing (alignment, mirror, loop close)."""

import numpy as np

from mjlab.tasks.pupper_gait import capture_jump as cj
from mjlab.tasks.pupper_gait import capture_reference as cap
from mjlab.tasks.pupper_gait.mdp import gait_reference

N = 100
DT = 0.02
DURATION = 0.75


def test_table_times_wrap_like_the_slot_lookup():
  grid = cj.table_times(N, DURATION)
  assert grid[0] == 0.0
  # Row k plays at phase k / N; phase 1 wraps to row 0, so no row sits at the end.
  np.testing.assert_allclose(np.diff(grid), DURATION / N)
  assert grid[-1] < DURATION


def test_align_indexes_at_the_reward_rewind():
  grid = cj.table_times(N, DURATION)
  # A fractional slot start (0.75 s grid at 50 Hz): samples land between steps.
  t_in = np.arange(-10, 60) * DT + 0.01
  q = np.tile(t_in[:, None], (1, 12))  # each joint reads its own sample time
  table = cj.align_jump(t_in, q, grid, DT, 1.8)
  # The row at reference time g holds what the joints did 1.8 steps later.
  np.testing.assert_allclose(table[:, 0], grid + 1.8 * DT, atol=1e-9)


def test_pronk_is_its_own_mirror_at_zero_shift():
  pronk = gait_reference.build_joint_reference_table(N, "jump")
  assert cap.mirror_shift(pronk) == 0
  # Exact up to the IK solver's tolerance.
  np.testing.assert_allclose(cap.mirror_symmetrize(pronk, 0), pronk, atol=1e-4)


def test_close_eases_the_tail_into_row_zero():
  rng = np.random.default_rng(0)
  table = rng.normal(size=(N, 12))
  closed = cj.close_to_start(table, 0.2)
  np.testing.assert_array_equal(closed[:80], table[:80])
  np.testing.assert_allclose(closed[-1], table[0])
  # The wrap from the last row back to row 0 is now continuous.
  assert np.abs(closed[-1] - closed[0]).max() < 1e-12
