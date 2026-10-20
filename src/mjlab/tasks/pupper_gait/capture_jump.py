"""Capture a trained Mystery policy's jump and install it as the jump-slot reference.

The one-shot counterpart of :mod:`capture_reference`: instead of folding many
cycles of a periodic gait, it averages many single jumps into one table.

1. **Rollout.** ``play=True`` Mystery env, command pinned to ``(speed, 0, 0)``.
   The random jump schedule is replaced by a fixed one -- slots on the gait
   grid every ``separation_s`` after ``warmup_s`` -- so every env jumps at
   known times. Envs that reset are dropped.
2. **Align.** Each jump is cut out on its own slot clock and indexed at the
   reward's rewound time (``t_in - GAIT_REWARD_PHASE_LEAD_STEPS * dt``), the
   same rewind ``mixed_jump_tracking`` applies, so the retrained reward grades
   the policy against exactly what it did. Slot starts are fractional steps
   (0.75 s grid at 50 Hz), so each jump is interpolated onto the table grid.
3. **Average.** Mean over the kept jumps (optionally only those with a real
   four-foot flight). The spread is reported as dispersion.
4. **Mirror.** A jump pushes off with both sides together, so the mirror shift
   is 0: ``L(t) = -R(t)``. Each left/right pair is replaced by its average.
5. **Close.** The slot plays row 0 at its start and wraps back to row 0 at
   its end, so the last ``close_frac`` of the table is eased into row 0.
6. **Install.** ``mdp/captures/jump_capture_round{k}.npz`` (archive) and
   ``mdp/captured_jump_gait.npz`` (keys ``jump``, ``duration_s``,
   ``touchdown_s``). Restart the session before retraining: the slot timing
   constants are read at import.

Run (GPU; the checkpoint must have trained against the jump reference that is
installed now, since the policy observes it during capture):

  python -m mjlab.tasks.pupper_gait.capture_jump \\
      --wandb-run-path <entity>/mjlab/<run-id> --round-index 1

``--task Mjlab-JumpOnly-Flat-Pupper-v3`` captures the jump-only task instead:
one jump per episode at its fixed launch time, command zero (``--speed`` and
``--jumps`` are ignored).
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
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import JOINT_NAMES
from mjlab.tasks.pupper.mdp.velocity_command import PupperVelocityCommand
from mjlab.tasks.pupper_gait.capture_reference import (
  _resolve_checkpoint,
  mirror_error,
  mirror_symmetrize,
  roll_lean,
  table_stats,
)
from mjlab.tasks.pupper_gait.mdp import gait_reference
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
  GAIT_N_SAMPLES,
  GAIT_REWARD_PHASE_LEAD_STEPS,
)

##
# Processing (pure NumPy, unit-tested).
##


def table_times(n_samples: int, duration_s: float) -> np.ndarray:
  """Slot time of each table row: row k plays at phase k / n (phase 1 wraps)."""
  return np.arange(n_samples) / n_samples * duration_s


def align_jump(
  t_in: np.ndarray,
  q: np.ndarray,
  grid: np.ndarray,
  step_dt: float,
  lead_steps: float,
) -> np.ndarray:
  """Resample one jump's ``(T, 12)`` joints onto the table ``grid`` [s].

  ``t_in`` is each sample's time since its slot start. Samples are indexed at
  the reward's rewound time, so the row at ``grid[k]`` holds what the policy
  did ``lead_steps`` steps after the reference showed ``grid[k]``.
  """
  tau = t_in - lead_steps * step_dt
  return np.stack([np.interp(grid, tau, q[:, j]) for j in range(q.shape[1])], axis=1)


def close_to_start(table: np.ndarray, close_frac: float) -> np.ndarray:
  """Ease the last ``close_frac`` of a one-shot table into its row 0."""
  n = len(table)
  k0 = int(round((1.0 - close_frac) * n))
  out = table.copy()
  if k0 >= n:
    return out
  u = (np.arange(k0, n) - k0 + 1) / (n - k0)
  w = (u * u * (3.0 - 2.0 * u))[:, None]
  out[k0:] = (1.0 - w) * table[k0:] + w * table[0]
  return out


##
# Rollout (GPU sim).
##


def rollout(
  task: str,
  checkpoint: Path,
  speed: float,
  num_envs: int,
  warmup_s: float,
  separation_s: float,
  jumps: int,
  window_s: float,
  device: str,
  seed: int,
) -> dict[str, Any]:
  """Run the policy with a fixed jump schedule; return per-jump recordings."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.pupper_gait.mdp.mixed_jump import _MAX_JUMPS, _slot_starts
  from mjlab.tasks.pupper_gait.mystery_env_cfg import JUMP_SLOT_S, JUMP_STAND_HEIGHT
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(task, play=True)
  # The jump-only task has no slot schedule: one jump per episode, launched
  # after the hold, with the command pinned at zero.
  slotted = "reset_jump_schedule" in env_cfg.events
  if not slotted:
    from mjlab.tasks.pupper_gait.jump_only_env_cfg import JUMP_ONLY_HOLD_S

    speed, jumps = 0.0, 1
  assert jumps <= _MAX_JUMPS, f"at most {_MAX_JUMPS} slots per episode"
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
  sensor = base.scene["feet_ground_contact"]
  assert tuple(robot.joint_names) == JOINT_NAMES, robot.joint_names
  dt = float(base.step_dt)

  twist = base.command_manager.get_term("twist")
  assert isinstance(twist, PupperVelocityCommand), type(twist)
  target = torch.tensor([speed, 0.0, 0.0], device=device).expand(num_envs, 3)
  update = twist._update_command

  def _hold() -> None:
    update()
    twist.vel_command_b[:] = target

  twist._update_command = _hold  # type: ignore[method-assign]

  if slotted:
    # Fixed schedule on the gait grid, replacing the random reset-time one.
    first = np.ceil(warmup_s / JUMP_SLOT_S) * JUMP_SLOT_S
    sep = np.ceil(separation_s / JUMP_SLOT_S) * JUMP_SLOT_S
    starts = first + sep * np.arange(jumps)
  else:
    starts = np.array([JUMP_ONLY_HOLD_S])
  n_steps = int(np.ceil((starts[-1] + window_s) / dt)) + 2

  alive = torch.ones(num_envs, dtype=torch.bool, device=device)
  q, t, z, flight = [], [], [], []
  with torch.no_grad():
    obs, _ = env.reset()
    if slotted:
      buf = _slot_starts(base)
      buf[:] = float("inf")
      buf[:, :jumps] = torch.tensor(starts, device=device, dtype=buf.dtype)
    for _ in range(n_steps):
      obs, _, _, _ = env.step(policy(obs))
      alive &= ~base.reset_buf
      q.append(robot.data.joint_pos.clone())
      t.append(base.episode_length_buf.float() * dt)
      z.append(robot.data.root_link_pos_w[:, 2].clone())
      flight.append((sensor.data.current_air_time > 0.0).all(dim=1))
  env.close()

  keep = alive.cpu().numpy()
  if not keep.any():
    raise RuntimeError("Every env reset during the capture.")
  return {
    "q": torch.stack(q).cpu().numpy()[:, keep],  # (T, E, 12)
    "t": torch.stack(t).cpu().numpy()[:, keep],  # (T, E) episode time
    "z": torch.stack(z).cpu().numpy()[:, keep],
    "flight": torch.stack(flight).cpu().numpy()[:, keep],
    "starts": starts,
    "step_dt": dt,
    "stand_height": JUMP_STAND_HEIGHT,
    "kept_envs": int(keep.sum()),
    "speed": speed,
  }


def main(
  wandb_run_path: str | None = None,
  checkpoint_name: str | None = None,
  checkpoint_file: str | None = None,
  round_index: int = 1,
  task: str = "Mjlab-Mystery-Flat-Pupper-v3",
  speed: float = 0.3,
  num_envs: int = 64,
  warmup_s: float = 2.0,
  separation_s: float = 3.0,
  jumps: int = 3,
  require_flight: bool = False,
  close_frac: float = 0.2,
  symmetrize: bool = True,
  seed: int = 0,
  install: bool = True,
) -> None:
  """Capture the policy's jump, average, symmetrize, install captured_jump_gait.npz.

  Args:
    speed: Forward command held during the capture [m/s]. Below the 0.5 m/s
      fast-gait onset, so the slot cuts into the trot.
    require_flight: Keep only jumps where all four feet left the ground. Off
      by default: a weak policy may have no full flights to keep.
    close_frac: Fraction of the table at the end eased back into row 0.
    install: Overwrite ``mdp/captured_jump_gait.npz``. The archive under
      ``mdp/captures/`` is written either way.
  """
  from mjlab.tasks.pupper_gait.mystery_env_cfg import JUMP_PLAYBACK_S

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  checkpoint, source = _resolve_checkpoint(
    task, wandb_run_path, checkpoint_name, checkpoint_file
  )
  installed = gait_reference._CAPTURED_JUMP_FILE.exists()
  duration = JUMP_PLAYBACK_S
  print(
    f"source {source}\nslot reference during capture: "
    f"{'installed captured_jump_gait.npz' if installed else 'parametric jump'} "
    f"| playback {duration:.3f} s"
  )

  window_s = duration + 0.3  # past the playback: the landing
  rec = rollout(
    task,
    checkpoint,
    speed,
    num_envs,
    warmup_s,
    separation_s,
    jumps,
    window_s,
    device,
    seed,
  )
  speed = rec["speed"]  # the jump-only task pins it to zero
  dt = rec["step_dt"]
  grid = table_times(GAIT_N_SAMPLES, duration)

  tables, heights, flights, touchdowns = [], [], [], []
  for e in range(rec["kept_envs"]):
    for start in rec["starts"]:
      t_in = rec["t"][:, e] - start
      sel = (t_in >= -0.2) & (t_in <= window_s)
      fl = rec["flight"][sel, e]
      t_sel = t_in[sel]
      airborne = fl & (t_sel >= 0.0)
      flew = bool(airborne.any())
      if require_flight and not flew:
        continue
      tables.append(
        align_jump(t_sel, rec["q"][sel, e], grid, dt, GAIT_REWARD_PHASE_LEAD_STEPS)
      )
      in_slot = (t_sel >= 0.0) & (t_sel <= duration)
      heights.append(float(rec["z"][sel, e][in_slot].max() - rec["stand_height"]))
      flights.append(float(airborne.sum() * dt))
      if flew:
        after = np.nonzero(airborne)[0][-1] + 1
        if after < len(t_sel):
          touchdowns.append(float(t_sel[after]))

  total = rec["kept_envs"] * len(rec["starts"])
  if not tables:
    raise RuntimeError(
      f"No jump kept out of {total}: none had a four-foot flight. "
      "Rerun with --require-flight False to capture the motion anyway."
    )
  stack = np.stack(tables)  # (J, N, 12)
  raw = stack.mean(axis=0)
  dispersion = 1000 * np.sqrt(np.mean((stack - raw) ** 2, axis=(0, 1)))
  table = mirror_symmetrize(raw, 0) if symmetrize else raw
  table = close_to_start(table, close_frac)
  flew_frac = float(np.mean(np.array(flights) > 0.0))
  # Touchdown as the reward sees it (rewound), clamped inside the playback so
  # the slot's active window stays shorter than its reward window.
  if touchdowns:
    touchdown = float(np.mean(touchdowns)) - GAIT_REWARD_PHASE_LEAD_STEPS * dt
    touchdown = float(np.clip(touchdown, 0.1, duration))
  else:
    touchdown = duration
  stats = table_stats(table, duration, dt)

  print(
    f"\n{len(tables)}/{total} jumps kept ({rec['kept_envs']} envs, "
    f"{speed:+.2f} m/s)\n"
    f"  four-foot flight in {100 * flew_frac:.0f}% | mean flight "
    f"{1000 * np.mean(flights):.0f} ms | mean peak rise "
    f"{100 * np.mean(heights):.1f} cm above stand\n"
    f"  touchdown {touchdown:.3f} s"
    f"{' (measured)' if touchdowns else ' (no flights: playback end)'}\n"
    f"  dispersion rms {dispersion.mean():.1f} mrad (worst joint "
    f"{dispersion.max():.1f})\n"
    f"  mirror error {mirror_error(raw, 0):.3f} -> {mirror_error(table, 0):.3f} rad "
    f"| roll lean {roll_lean(raw):.3f} -> {roll_lean(table):.3f} rad\n"
    f"  worst move {stats['rad_per_step']:.3f} rad/step -> {stats['torque']:.2f} Nm "
    f"| at joint limit {100 * stats['at_limit']:.1f}%"
  )

  out: dict[str, Any] = {
    "jump": table,
    "jump_raw": raw,
    "duration_s": np.array(duration),
    "touchdown_s": np.array(touchdown),
    "meta_source": np.array(source),
    "meta_round": np.array(round_index),
    "meta_speed": np.array(speed),
    "meta_jumps_kept": np.array(len(tables)),
    "meta_jumps_total": np.array(total),
    "meta_flight_frac": np.array(flew_frac),
    "meta_mean_flight_s": np.array(np.mean(flights)),
    "meta_mean_peak_rise_m": np.array(np.mean(heights)),
    "meta_dispersion_mrad": dispersion,
    "meta_require_flight": np.array(require_flight),
    "meta_symmetrized": np.array(symmetrize),
    "meta_close_frac": np.array(close_frac),
    "meta_phase_lead_steps": np.array(GAIT_REWARD_PHASE_LEAD_STEPS),
  }
  archive = gait_reference._CAPTURED_JUMP_FILE.parent / "captures"
  archive.mkdir(exist_ok=True)
  archived = archive / f"jump_capture_round{round_index}.npz"
  np.savez(archived, **out)
  print(f"\narchived {archived}")
  if install:
    shutil.copyfile(archived, gait_reference._CAPTURED_JUMP_FILE)
    print(
      f"installed {gait_reference._CAPTURED_JUMP_FILE}\n"
      "Restart the session before retraining (slot timing is read at import)."
    )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
