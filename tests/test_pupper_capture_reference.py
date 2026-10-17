"""Tests for the reach-gait capture pipeline (processing and the loader socket)."""

import numpy as np
import pytest

from mjlab.tasks.pupper_gait import capture_reference as cap
from mjlab.tasks.pupper_gait.mdp import gait_reference

N = 100


def _smooth_cycle(rng: np.random.Generator, harmonics: int = 3) -> np.ndarray:
  """A random periodic (N, 12) cycle made of a few harmonics."""
  grid = np.arange(N) / N
  basis = cap._fourier_basis(grid, harmonics)
  return basis @ rng.normal(scale=0.2, size=(basis.shape[1], 12))


@pytest.fixture
def isolated_capture_socket(tmp_path, monkeypatch):
  monkeypatch.setattr(gait_reference, "_CAPTURED_GAITS_FILE", tmp_path / "cap.npz")
  monkeypatch.setattr(gait_reference, "_TABLE_CACHE", {})
  return tmp_path / "cap.npz"


def test_reference_phase_matches_reward_rewind():
  steps = np.arange(200, dtype=np.float64)
  phase = cap.reference_phase(steps, 0.02, 8 / 3, 1.8)
  expected = np.mod((steps - 1.8) * 0.02 * 8 / 3, 1.0)
  np.testing.assert_allclose(phase, expected)
  assert phase.min() >= 0.0 and phase.max() < 1.0


def test_fold_recovers_cycle_from_sparse_noisy_samples():
  rng = np.random.default_rng(0)
  truth = _smooth_cycle(rng)
  # 50 Hz on a 0.375 s cycle: only 75 distinct phases, like a real capture.
  steps = np.arange(2000, dtype=np.float64)
  phase = cap.reference_phase(steps, 0.02, 8 / 3, 1.8)
  grid = np.arange(N) / N
  clean = np.stack(
    [np.interp(phase, np.append(grid, 1.0), np.append(truth[:, j], truth[0, j]))
     for j in range(12)],
    axis=1,
  )  # fmt: skip
  noisy = clean + rng.normal(scale=0.01, size=clean.shape)
  table, resid = cap.fold_cycle(noisy, phase, N, harmonics=8)
  assert table.shape == (N, 12)
  assert np.abs(table - truth).max() < 0.01
  # Residual is the injected noise, not structure.
  assert abs(np.sqrt(np.mean(resid**2)) - 0.01) < 0.002


def test_fold_is_periodic_at_wrap():
  rng = np.random.default_rng(1)
  phase = rng.uniform(size=3000)
  q = rng.normal(size=(3000, 12))
  table, _ = cap.fold_cycle(q, phase, N, harmonics=8)
  step = np.abs(np.diff(table, axis=0)).max()
  assert np.abs(table[0] - table[-1]).max() <= step + 1e-9


def test_mirror_shift_finds_trot_half_cycle():
  for gait in ("trot", "reach"):
    table = gait_reference.build_joint_reference_table(N, gait)
    assert cap.mirror_shift(table) == N // 2


def test_mirror_symmetrize_removes_asymmetry_and_lean():
  rng = np.random.default_rng(2)
  right = _smooth_cycle(rng)
  sym = right.copy()
  for r, left in cap._PAIRS:
    sym[:, left : left + 3] = -np.roll(right[:, r : r + 3], 30, 0)
  # Symmetric input is a fixed point.
  np.testing.assert_allclose(cap.mirror_symmetrize(sym, 30), sym, atol=1e-12)
  # A sustained lean on one leg's roll joint is averaged out.
  leaned = sym.copy()
  leaned[:, 3 + cap._ROLL] += 0.3
  assert cap.roll_lean(leaned) > 0.25
  fixed = cap.mirror_symmetrize(leaned, cap.mirror_shift(leaned))
  assert cap.mirror_error(fixed, 30) < 1e-12
  assert cap.roll_lean(fixed) < 1e-9


def test_installed_capture_is_what_the_loader_serves(isolated_capture_socket):
  rng = np.random.default_rng(3)
  fwd, back = _smooth_cycle(rng), _smooth_cycle(rng)
  np.savez(isolated_capture_socket, reach_fwd=fwd, reach_back=back, meta_round=1)
  np.testing.assert_allclose(
    gait_reference.build_joint_reference_table(N, "reach"), fwd
  )
  served_back = gait_reference.build_joint_reference_table(N, "reach_back")
  # The backward pipeline reads the table at eff_phase = -phase; that must
  # return the capture at the forward-running phase it was recorded at.
  for i in range(N):
    np.testing.assert_allclose(served_back[(-i) % N], back[i])
