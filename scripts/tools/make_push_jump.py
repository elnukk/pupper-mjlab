"""Hand-built starter reference for the Mystery task's jump slot.

The parametric pronk (all legs in phase) only tucks the feet: with the feet on
the ground that lowers the body, so a policy tracking it learns to crouch, not
to jump. This table adds the missing push: stand, crouch, fast extend (the
launch), tuck in the air, reach down, settle back to stand. All four feet move
together, directly under their hips.

It is a starter to coax a jump out of the policy, not the final reference --
capture the jump the policy then does and install that instead.

Writes mdp/captures/jump_push_v1.npz and installs it as
mdp/captured_jump_gait.npz (keys: jump (N, 12), duration_s, touchdown_s).

  uv run python scripts/tools/make_push_jump.py
"""

from __future__ import annotations

import shutil
from typing import Any

import numpy as np

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  JOINT_NAMES,
  PUPPER_EFFORT_LIMIT,
  PUPPER_KP,
  get_spec,
)
from mjlab.tasks.pupper_gait.mdp import gait_reference as gr

N = 100
DURATION_S = 0.75  # one base gait cycle, the slot grid
CONTROL_DT = 0.02

# (time [s], foot depth below the hip [m]). Stand depth matches the trot stance.
KEYFRAMES = (
  (0.00, -0.14),  # stand
  # Crouch as deep as the belly allows (2.4 cm gap, outside torso_clearance's
  # 2 cm zone) and extend to near full reach (IK degrades past -0.165); the
  # push takes 0.1 s, which keeps it under the 3 Nm motor limit (2.81 Nm).
  (0.15, -0.07),  # crouch: load the legs
  (0.25, -0.165),  # fast extend: the push-off
  (0.45, -0.07),  # tuck in the air
  (0.60, -0.14),  # reach down: touchdown
  (0.68, -0.12),  # absorb the landing
  (0.75, -0.14),  # settle back to stand (= row 0, so the playback closes)
)
TOUCHDOWN_S = 0.60


def foot_depth(t: np.ndarray) -> np.ndarray:
  """Smoothstep between keyframes: zero foot velocity at every keyframe."""
  ts = np.array([k[0] for k in KEYFRAMES])
  zs = np.array([k[1] for k in KEYFRAMES])
  i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2)
  u = (t - ts[i]) / (ts[i + 1] - ts[i])
  s = u * u * (3.0 - 2.0 * u)
  return zs[i] + (zs[i + 1] - zs[i]) * s


def build_table() -> tuple[np.ndarray, float]:
  # Non-periodic grid: row 0 is t=0, row N-1 is the end (the loader resamples
  # one-shot captures endpoint to endpoint).
  t = np.linspace(0.0, DURATION_S, N)
  z = foot_depth(t)
  table = np.zeros((N, 12))
  guesses = [np.zeros(3) for _ in range(4)]
  worst_ik = 0.0
  for i in range(N):
    for leg in range(4):
      target = gr._HIP_OFFSET[leg] + np.array([0.0, 0.0, z[i]])
      theta = gr._ik_single_leg(target, leg, guesses[leg])
      table[i, 3 * leg : 3 * leg + 3] = theta
      guesses[leg] = theta
      worst_ik = max(worst_ik, float(np.linalg.norm(gr._FK[leg](theta) - target)))
  return table, worst_ik


def main() -> None:
  table, worst_ik = build_table()
  model = get_spec().compile()
  limits = np.array([model.joint(n).range for n in JOINT_NAMES])
  at_limit = np.mean((table <= limits[:, 0] + 1e-3) | (table >= limits[:, 1] - 1e-3))
  per_step = np.abs(np.diff(table, axis=0)).max() * (
    CONTROL_DT / (DURATION_S / (N - 1))
  )
  torque = PUPPER_KP * per_step
  flag = "  OVER LIMIT" if torque > PUPPER_EFFORT_LIMIT else ""
  print(f"ik residual   {1000 * worst_ik:.1f} mm")
  print(f"at limit      {100 * at_limit:.1f}%")
  print(f"rad/step      {per_step:.3f}  (torque {torque:.2f} Nm{flag})")
  print(f"closes        {np.abs(table[0] - table[-1]).max():.4f} rad end-to-start")

  out: dict[str, Any] = {
    "jump": table,
    "duration_s": np.array(DURATION_S),
    "touchdown_s": np.array(TOUCHDOWN_S),
    "meta_source": np.array("hand-built push reference (make_push_jump.py)"),
    "meta_keyframes": np.array(KEYFRAMES),
  }
  archive = gr._CAPTURED_JUMP_FILE.parent / "captures"
  archive.mkdir(exist_ok=True)
  archived = archive / "jump_push_v1.npz"
  np.savez(archived, **out)
  shutil.copyfile(archived, gr._CAPTURED_JUMP_FILE)
  print(f"archived  {archived}\ninstalled {gr._CAPTURED_JUMP_FILE}")


if __name__ == "__main__":
  main()
