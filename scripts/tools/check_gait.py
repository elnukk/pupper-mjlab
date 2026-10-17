"""Score a parametric gait design before training it.

Prints, per design: the no-slip body speed the footprint implies, the worst
joint move per 20 ms control step (and the motor torque that move needs),
the IK residual, and how often the table hits a joint limit.

Edit DESIGNS below, then run:
  uv run python scripts/tools/check_gait.py
"""

from __future__ import annotations

import numpy as np

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  JOINT_NAMES,
  PUPPER_EFFORT_LIMIT,
  PUPPER_KP,
  get_spec,
)
from mjlab.tasks.pupper_gait.mdp import gait_reference as gr
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import GAIT_CYCLE_S

CONTROL_DT = 0.02
N = 100

# Each design: touchdown (FR, FL, BR, LB), stride, stance_z, swing_lift, and
# optionally swing_kf, freq_mult, x_shift, y_shift.
DESIGNS: dict[str, dict] = {
  "trot (shipped)": dict(
    touchdown=(0.0, 0.5, 0.5, 0.0), stride=0.05, stance_z=-0.14, swing_lift=0.09
  ),
  "reach (in repo)": dict(
    touchdown=gr._GAIT_TOUCHDOWN["reach"],
    stride=gr._GAIT_STRIDE["reach"],
    stance_z=gr._GAIT_STANCE_Z["reach"],
    swing_lift=gr._GAIT_SWING_LIFT["reach"],
    swing_kf=gr._GAIT_SWING_KF["reach"],
    freq_mult=gr._GAIT_FREQ_MULT["reach"],
  ),
  "trot, long + 2x": dict(
    touchdown=(0.0, 0.5, 0.5, 0.0),
    stride=0.09,
    stance_z=-0.12,
    swing_lift=0.06,
    freq_mult=2.0,
  ),
  "trot, long + 2x + swing3": dict(
    touchdown=(0.0, 0.5, 0.5, 0.0),
    stride=0.09,
    stance_z=-0.12,
    swing_lift=0.06,
    swing_kf=3,
    freq_mult=2.0,
  ),
  "cheetah order, swing5, 2.5x": dict(
    touchdown=(0.1, 0.0, 0.6, 0.5),
    stride=0.09,
    stance_z=-0.12,
    swing_lift=0.06,
    swing_kf=5,
    freq_mult=2.5,
  ),
}


def _register(name: str, d: dict) -> None:
  gr._GAIT_TOUCHDOWN[name] = d["touchdown"]
  gr._GAIT_STRIDE[name] = d["stride"]
  gr._GAIT_STANCE_Z[name] = d["stance_z"]
  gr._GAIT_SWING_LIFT[name] = d["swing_lift"]
  gr._GAIT_SWING_KF[name] = d.get("swing_kf", 1)
  for table, key in ((gr._GAIT_X_SHIFT, "x_shift"), (gr._GAIT_Y_SHIFT, "y_shift")):
    if key in d:
      table[name] = d[key]
    else:
      table.pop(name, None)


def _ik_residual(name: str, d: dict, table: np.ndarray) -> float:
  cycle = gr._base_cycle(
    d["stride"], d.get("swing_kf", 1), d["stance_z"], d["swing_lift"]
  )
  xs = d.get("x_shift", (0.0,) * 4)
  ys = d.get("y_shift", (0.0,) * 4)
  worst = 0.0
  for i in range(N):
    for leg in range(4):
      foot = gr._interpolate_triangle((i / N - d["touchdown"][leg]) % 1.0, cycle)
      target = foot + gr._HIP_OFFSET[leg] + np.array([xs[leg], ys[leg], 0.0])
      got = gr._FK[leg](table[i, 3 * leg : 3 * leg + 3])
      worst = max(worst, float(np.linalg.norm(got - target)))
  return worst


def score(name: str, d: dict, limits: np.ndarray) -> dict:
  key = f"_check_{name}"
  _register(key, d)
  table = gr.build_joint_reference_table(N, key)
  swing_kf = d.get("swing_kf", 1)
  duty = 4 / (5 + swing_kf)
  cycle_s = GAIT_CYCLE_S / d.get("freq_mult", 1.0)
  stride = d["stride"]
  span = sum(stride) if isinstance(stride, tuple) else 2 * stride
  speed = span / (duty * cycle_s)

  # Joint move per control step: table rows are 1/N cycle apart.
  steps_per_cycle = cycle_s / CONTROL_DT
  diffs = np.abs(np.diff(np.vstack([table, table[:1]]), axis=0))
  per_step = diffs.max() * N / steps_per_cycle
  at_limit = np.mean((table <= limits[:, 0] + 1e-3) | (table >= limits[:, 1] - 1e-3))
  # Feet down at once, averaged over the cycle.
  feet_down = 4 * duty
  return dict(
    speed=speed,
    duty=duty,
    cycle_s=cycle_s,
    feet_down=feet_down,
    per_step=per_step,
    torque=PUPPER_KP * per_step,
    ik_mm=1000 * _ik_residual(key, d, table),
    at_limit=at_limit,
  )


def main() -> None:
  model = get_spec().compile()
  limits = np.array([model.joint(n).range for n in JOINT_NAMES])
  print(
    f"{'design':30s} {'speed':>6s} {'duty':>5s} {'cycle':>6s} {'feet':>5s} "
    f"{'rad/step':>8s} {'torque':>7s} {'ik_mm':>6s} {'@limit':>7s}"
  )
  for name, d in DESIGNS.items():
    s = score(name, d, limits)
    flag = " !" if s["torque"] > PUPPER_EFFORT_LIMIT else ""
    print(
      f"{name:30s} {s['speed']:6.2f} {s['duty']:5.2f} {s['cycle_s']:6.3f} "
      f"{s['feet_down']:5.2f} {s['per_step']:8.3f} {s['torque']:6.2f}{flag:1s} "
      f"{s['ik_mm']:6.1f} {100 * s['at_limit']:6.1f}%"
    )
  print(
    f"\nspeed = no-slip m/s the footprint implies (target 1.5); torque = kp x "
    f"rad/step, '!' = over the {PUPPER_EFFORT_LIMIT} Nm limit; ik_mm under ~2 "
    "is clean; @limit should be ~0%."
  )


if __name__ == "__main__":
  main()
