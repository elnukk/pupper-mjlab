"""Capture a trained policy's fast gait and install it as the reach reference.

Reference bootstrapping: roll out a trained MixedGaits policy with the command
held at +v and -v, record its joint trajectories, fold them into one cycle,
mirror-symmetrize left/right, and write ``mdp/captured_reach_gaits.npz`` -- the
socket ``gait_reference.build_joint_reference_table`` reads for ``reach`` and
``reach_back``. Retrain (warm-started) on it, capture again, repeat.

Pipeline and the reasoning behind each choice:

1. **Rollout.** ``play=True`` env (no pushes, no timeouts, observation noise,
   latency and per-episode DR still on), command pinned to ``(v, 0, 0)``. The
   first ``warmup_s`` are discarded (getting up to speed), and any env that
   resets at any point is dropped entirely: steady state only.
2. **Phase.** Each sample is indexed by the *reference clock* the policy saw,
   rewound by ``GAIT_REWARD_PHASE_LEAD_STEPS`` -- the same rewind the
   ``gait_tracking`` reward applies. The joints lag the observed reference by
   the pipeline delay; indexing at the rewound phase means the retrained
   reward grades the policy against exactly what it did, instead of shifting
   the reference by one delay per round. No cross-correlation alignment is
   needed: the policy is phase-locked to the clock it observes, which the
   reported dispersion verifies.
3. **Fold.** A least-squares Fourier fit with ``harmonics`` terms per joint.
   Periodic by construction (the wrap point is continuous in value and
   slope), smooths per-step noise, and evaluates directly on the 100-sample
   grid. The policy runs at 50 Hz on a 0.375 s cycle, so a single env only
   ever visits ~75 distinct phases; the fit fills the grid without empty bins.
4. **Mirror.** Left legs are the right legs negated and phase-shifted
   (``L(t) = -R(t - s)``; exact for the parametric gaits, ``s = 0.5`` for the
   trot pairing). ``s`` is estimated from the capture (one shift shared by
   the front and back pairs), then each pair is replaced by its symmetric
   average. This removes the leg-roll lean the ``roll_asymmetry`` reward
   penalizes, so the reward defends the edit instead of fighting the
   reference.

Tables are stored as absolute joint angles in ``JOINT_NAMES`` order, both
indexed by the forward-running phase; the loader pre-reverses ``reach_back``
for the backward pipeline. The pre-mirror tables and capture statistics are
saved alongside (``*_raw``, ``meta_*``) for the per-round report.

Run (GPU; the checkpoint must have trained against the reference currently in
the repo, since the policy observes it during capture):

  uv run python -m mjlab.tasks.pupper_gait.capture_reference \\
      --wandb-run-path <entity>/mjlab/<run-id> --round-index 1
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  JOINT_NAMES,
  PUPPER_KP,
  get_spec,
)
from mjlab.tasks.pupper.mdp.velocity_command import PupperVelocityCommand
from mjlab.tasks.pupper_gait.mdp import gait_reference
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
  GAIT_FREQUENCY,
  GAIT_N_SAMPLES,
  GAIT_REWARD_PHASE_LEAD_STEPS,
)

# (right, left) first-joint column of each mirrored leg pair: FR/FL, BR/LB.
_PAIRS: tuple[tuple[int, int], ...] = ((0, 3), (6, 9))
# Leg-roll (motor 2) column within a leg -- the joint roll_asymmetry watches.
_ROLL = 1


##
# Processing (pure NumPy, unit-tested).
##


def reference_phase(
  steps: np.ndarray, step_dt: float, frequency: float, lead_steps: float
) -> np.ndarray:
  """Phase in [0, 1) of the reward's (rewound) reference clock at each step."""
  return np.mod((steps - lead_steps) * step_dt * frequency, 1.0)


def _fourier_basis(phase: np.ndarray, harmonics: int) -> np.ndarray:
  cols = [np.ones_like(phase)]
  for k in range(1, harmonics + 1):
    cols += [np.cos(2 * np.pi * k * phase), np.sin(2 * np.pi * k * phase)]
  return np.stack(cols, axis=1)


def fold_cycle(
  q: np.ndarray, phase: np.ndarray, n_samples: int, harmonics: int
) -> tuple[np.ndarray, np.ndarray]:
  """Fit one periodic cycle to ``(S, 12)`` samples at ``(S,)`` phases.

  Returns:
    The ``(n_samples, 12)`` table on the grid ``i / n_samples``, and the
    ``(S, 12)`` per-sample residual (sample minus fitted cycle).
  """
  basis = _fourier_basis(phase, harmonics)
  coef, *_ = np.linalg.lstsq(basis, q, rcond=None)
  grid = np.arange(n_samples) / n_samples
  table = _fourier_basis(grid, harmonics) @ coef
  return table, q - basis @ coef


def mirror_shift(table: np.ndarray) -> int:
  """Grid shift ``s`` that best satisfies ``L(t) = -R(t - s)`` for both pairs."""
  n = len(table)
  costs = [
    sum(
      float(
        np.sum((table[:, left : left + 3] + np.roll(table[:, r : r + 3], s, 0)) ** 2)
      )
      for r, left in _PAIRS
    )
    for s in range(n)
  ]
  return int(np.argmin(costs))


def mirror_symmetrize(table: np.ndarray, shift: int) -> np.ndarray:
  """Replace each left/right pair by its mirror average under ``shift``."""
  out = table.copy()
  for r, left in _PAIRS:
    right = 0.5 * (table[:, r : r + 3] - np.roll(table[:, left : left + 3], -shift, 0))
    out[:, r : r + 3] = right
    out[:, left : left + 3] = -np.roll(right, shift, 0)
  return out


def mirror_error(table: np.ndarray, shift: int) -> float:
  """Worst ``|L(t) + R(t - s)|`` over both pairs and all joints [rad]."""
  return max(
    float(
      np.abs(table[:, left : left + 3] + np.roll(table[:, r : r + 3], shift, 0)).max()
    )
    for r, left in _PAIRS
  )


def roll_lean(table: np.ndarray) -> float:
  """Worst cycle-mean leg-roll pair sum [rad]: the DC lean roll_asymmetry sees."""
  return max(
    abs(float(np.mean(table[:, r + _ROLL] + table[:, left + _ROLL])))
    for r, left in _PAIRS
  )


def table_stats(table: np.ndarray, cycle_s: float, step_dt: float) -> dict:
  """Feasibility numbers in the same terms as scripts/tools/check_gait.py."""
  limits = np.array([get_spec().compile().joint(n).range for n in JOINT_NAMES])
  diffs = np.abs(np.diff(np.vstack([table, table[:1]]), axis=0))
  per_step = float(diffs.max()) * len(table) * step_dt / cycle_s
  at_limit = float(
    np.mean((table <= limits[:, 0] + 1e-3) | (table >= limits[:, 1] - 1e-3))
  )
  return {
    "rad_per_step": per_step,
    "torque": PUPPER_KP * per_step,
    "at_limit": at_limit,
  }


##
# Rollout (GPU sim).
##


def rollout(
  task: str,
  checkpoint: Path,
  speed: float,
  num_envs: int,
  warmup_s: float,
  capture_s: float,
  device: str,
  seed: int,
) -> dict[str, np.ndarray | float]:
  """Run the policy at a pinned ``(speed, 0, 0)`` command and record joints.

  Returns ``q`` ``(S, 12)`` and ``steps`` ``(S,)`` for the surviving envs'
  steady-state window, flattened, plus summary statistics.
  """
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  agent_cfg = load_rl_cfg(task)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions
  )
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
    env, asdict(agent_cfg), device=device
  )
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  base = env.unwrapped
  robot = base.scene["robot"]
  assert tuple(robot.joint_names) == JOINT_NAMES, robot.joint_names

  # Pin the command after every internal update (resample and heading hold).
  twist = base.command_manager.get_term("twist")
  assert isinstance(twist, PupperVelocityCommand), type(twist)
  target = torch.tensor([speed, 0.0, 0.0], device=device).expand(num_envs, 3)
  update = twist._update_command

  def _hold() -> None:
    update()
    twist.vel_command_b[:] = target

  twist._update_command = _hold  # type: ignore[method-assign]

  n_warmup = round(warmup_s / base.step_dt)
  n_capture = round(capture_s / base.step_dt)
  alive = torch.ones(num_envs, dtype=torch.bool, device=device)
  q, steps, vx = [], [], []
  with torch.no_grad():
    obs, _ = env.reset()
    for t in range(n_warmup + n_capture):
      obs, _, _, _ = env.step(policy(obs))
      alive &= ~base.reset_buf
      if t >= n_warmup:
        # Joints after the step, against the clock the reward reads them at.
        q.append(robot.data.joint_pos.clone())
        steps.append(base.episode_length_buf.clone())
        vx.append(robot.data.root_link_lin_vel_b[:, 0].clone())
  env.close()

  keep = alive.cpu().numpy()
  if not keep.any():
    raise RuntimeError(f"Every env reset during the {speed:+.2f} m/s capture.")
  q_arr = torch.stack(q).cpu().numpy()[:, keep]  # (T, E, 12)
  steps_arr = torch.stack(steps).cpu().numpy()[:, keep]  # (T, E)
  vx_arr = torch.stack(vx).cpu().numpy()[:, keep]
  return {
    "q": q_arr.reshape(-1, 12),
    "steps": steps_arr.reshape(-1).astype(np.float64),
    "step_dt": float(base.step_dt),
    "kept_envs": int(keep.sum()),
    "mean_vx": float(vx_arr.mean()),
  }


def _resolve_checkpoint(
  task: str, wandb_run_path: str | None, checkpoint_name: str | None, file: str | None
) -> tuple[Path, str]:
  if file is not None:
    return Path(file), file
  if wandb_run_path is None:
    raise ValueError("Pass --wandb-run-path or --checkpoint-file.")
  from mjlab.tasks.registry import load_rl_cfg
  from mjlab.utils.os import get_wandb_checkpoint_path

  log_root = (Path("logs/rsl_rl") / load_rl_cfg(task).experiment_name).resolve()
  path, _ = get_wandb_checkpoint_path(log_root, Path(wandb_run_path), checkpoint_name)
  return path, f"{wandb_run_path}/{path.name}"


def main(
  wandb_run_path: str | None = None,
  checkpoint_name: str | None = None,
  checkpoint_file: str | None = None,
  round_index: int = 1,
  task: str = "Mjlab-MixedGaits-Flat-Pupper-v3",
  speed: float = 1.5,
  num_envs: int = 64,
  warmup_s: float = 3.0,
  capture_s: float = 10.0,
  harmonics: int = 8,
  symmetrize: bool = True,
  seed: int = 0,
  install: bool = True,
) -> None:
  """Capture +-speed, fold, symmetrize, and install captured_reach_gaits.npz.

  Args:
    round_index: Bootstrapping round this capture feeds (1 = first capture,
      from the Part 1 policy). Also names the archived copy.
    install: Overwrite ``mdp/captured_reach_gaits.npz``. The archive under
      ``mdp/captures/`` is written either way.
  """
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  checkpoint, source = _resolve_checkpoint(
    task, wandb_run_path, checkpoint_name, checkpoint_file
  )
  frequency = GAIT_FREQUENCY * gait_reference._GAIT_FREQ_MULT.get("reach", 1.0)
  cycle_s = 1.0 / frequency
  print(f"source {source} | fast clock {frequency:.3f} Hz ({cycle_s:.3f} s)")

  out: dict[str, Any] = {}
  for key, v in (("reach_fwd", speed), ("reach_back", -speed)):
    rec = rollout(task, checkpoint, v, num_envs, warmup_s, capture_s, device, seed)
    step_dt = float(rec["step_dt"])
    phase = reference_phase(
      np.asarray(rec["steps"]), step_dt, frequency, GAIT_REWARD_PHASE_LEAD_STEPS
    )
    raw, resid = fold_cycle(np.asarray(rec["q"]), phase, GAIT_N_SAMPLES, harmonics)
    shift = mirror_shift(raw)
    table = mirror_symmetrize(raw, shift) if symmetrize else raw
    stats = table_stats(table, cycle_s, step_dt)
    dispersion = 1000 * np.sqrt(np.mean(resid**2, axis=0))
    print(
      f"\n{key} @ {v:+.2f} m/s: {rec['kept_envs']}/{num_envs} envs kept, "
      f"mean vx {rec['mean_vx']:+.3f}\n"
      f"  dispersion rms {dispersion.mean():.1f} mrad (worst joint "
      f"{dispersion.max():.1f}) -- spread of samples around the folded cycle\n"
      f"  mirror shift {shift / GAIT_N_SAMPLES:.2f} cycle | mirror error "
      f"{mirror_error(raw, shift):.3f} -> {mirror_error(table, shift):.3f} rad | "
      f"roll lean {roll_lean(raw):.3f} -> {roll_lean(table):.3f} rad\n"
      f"  worst move {stats['rad_per_step']:.3f} rad/step -> {stats['torque']:.2f} "
      f"Nm | at joint limit {100 * stats['at_limit']:.1f}%"
    )
    out[key] = table
    out[f"{key}_raw"] = raw
    out[f"meta_{key}_mean_vx"] = np.array(rec["mean_vx"])
    out[f"meta_{key}_dispersion_mrad"] = dispersion
    out[f"meta_{key}_mirror_shift"] = np.array(shift / GAIT_N_SAMPLES)
    out[f"meta_{key}_kept_envs"] = np.array(rec["kept_envs"])

  out["meta_source"] = np.array(source)
  out["meta_round"] = np.array(round_index)
  out["meta_speed"] = np.array(speed)
  out["meta_harmonics"] = np.array(harmonics)
  out["meta_symmetrized"] = np.array(symmetrize)
  out["meta_frequency_hz"] = np.array(frequency)
  out["meta_phase_lead_steps"] = np.array(GAIT_REWARD_PHASE_LEAD_STEPS)

  archive = gait_reference._CAPTURED_GAITS_FILE.parent / "captures"
  archive.mkdir(exist_ok=True)
  archived = archive / f"captured_reach_gaits_round{round_index}.npz"
  np.savez(archived, **out)
  print(f"\narchived {archived}")
  if install:
    shutil.copyfile(archived, gait_reference._CAPTURED_GAITS_FILE)
    print(f"installed {gait_reference._CAPTURED_GAITS_FILE}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
