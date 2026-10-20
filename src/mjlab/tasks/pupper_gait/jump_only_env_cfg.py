"""Jump-only task: one in-place jump per episode, nothing else to do.

The Mystery slot asks for a jump *inside* running, graded by a reward that is
mostly locomotion -- the jump terms were under 1% of it, and the policy
settled for a crouch. This task isolates the motion so it can be perfected
first and then captured as the slot reference (``capture_jump --task
Mjlab-JumpOnly-Flat-Pupper-v3``):

- zero command, short episodes: stand ``JUMP_ONLY_HOLD_S``, play the jump
  reference once (:mod:`mdp.jump`), land, stand;
- the locomotion terms that fight a jump (velocity tracking, base height,
  vertical-velocity penalty, air time, foot slip, stand-still) are removed,
  so a weight dict written for the gait tasks cannot switch them back on;
- the jump terms are gated to the jump window: up-velocity only from the
  launch to touchdown (one rise per episode, not repeated bobbing), airborne
  height only in a four-foot flight, sinking during the hold is penalized.

The reference is whatever ``mdp/captured_jump_gait.npz`` holds (install the
hand-built push, ``captures/jump_push_v1.npz``, to start), else the
parametric pronk. Same 48-dim actor frame and critic layout as the Mystery
task, so a Mystery checkpoint warm-starts it.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pupper.pupper_env_cfg import pupper_flat_env_cfg
from mjlab.tasks.pupper_gait import mdp
from mjlab.tasks.pupper_gait.mdp.gait_reference import (
  jump_capture_duration_s,
  jump_capture_touchdown_s,
)
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
  GAIT_N_SAMPLES,
  GAIT_REWARD_PHASE_LEAD_STEPS,
  GAIT_STD,
)

# Stand still this long after reset before the reference launches: the robot
# settles from the spawn and the strict hold std grades the stance.
JUMP_ONLY_HOLD_S: float = 0.5

# One playback of the jump reference: the capture's duration, else one base
# gait cycle (the parametric pronk).
_capture = jump_capture_duration_s()
JUMP_ONLY_PLAYBACK_S: float = _capture if _capture is not None else 0.75
_touchdown = jump_capture_touchdown_s()
JUMP_ONLY_TOUCHDOWN_S: float = (
  _touchdown if _touchdown is not None else JUMP_ONLY_PLAYBACK_S
)

# Hold, jump, then a second and a quarter to land and stand still again.
JUMP_ONLY_EPISODE_S: float = JUMP_ONLY_HOLD_S + JUMP_ONLY_PLAYBACK_S + 1.25

# Loose tracking std from launch to a little past the playback (the landing).
JUMP_ONLY_LOOSE_UNTIL_S: float = JUMP_ONLY_HOLD_S + JUMP_ONLY_PLAYBACK_S + 0.1

# Same stand height as the Mystery slot terms. The height cap is much lower:
# with 0.40 the first policy that flew (~8.5 airborne reward) bought height
# with a pitched rear-leg kick, because every extra centimeter outpaid the
# form penalties. Past 6 cm the airborne term pays nothing more, so the only
# reward left to earn is form.
JUMP_ONLY_STAND_HEIGHT: float = 0.145
JUMP_ONLY_HEIGHT_CAP: float = 0.06
JUMP_ONLY_TORSO_CLEARANCE: float = 0.02

# A jump pitches harder than a walk; 45 deg still catches real falls.
JUMP_ONLY_FELL_OVER_LIMIT_ANGLE: float = math.radians(45.0)

# Removed outright (see module docstring).
_LOCOMOTION_TERMS = (
  "track_linear_velocity",
  "base_height",
  "lin_vel_z_l2",
  "air_time",
  "foot_slip",
  "pose",
  "stand_still_pose",
  "stand_still_joint_velocity",
)


def pupper_jump_only_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """In-place single jump on flat ground (see module docstring)."""
  cfg = pupper_flat_env_cfg(play=play)
  # Play loops episodes too, so the viewer shows a jump every few seconds.
  cfg.episode_length_s = JUMP_ONLY_EPISODE_S
  cfg.terminations["fell_over"].params["limit_angle"] = JUMP_ONLY_FELL_OVER_LIMIT_ANGLE
  # No command ramp: the command is zero for the whole run.
  cfg.curriculum = {}

  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.PupperVelocityCommandCfg)
  twist.rel_standing_envs = 1.0
  twist.rel_forward_envs = 0.0
  twist.rel_turn_envs = 0.0
  twist.ranges.lin_vel_x = (0.0, 0.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)

  ref = {
    "frequency": 1.0 / JUMP_ONLY_PLAYBACK_S,
    "n_samples": GAIT_N_SAMPLES,
    "crouch_hold_s": JUMP_ONLY_HOLD_S,
  }
  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      "policy": ObservationTermCfg(
        func=mdp.jump_actor_obs,
        params={
          "add_noise": True,
          "imu_latency_distribution": mdp.PUPPER_IMU_LATENCY_DIST,
          **ref,
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=20,
  )
  cfg.observations["critic"].terms["gait_reference"] = ObservationTermCfg(
    func=mdp.jump_reference_obs, params=dict(ref)
  )

  r = cfg.rewards
  for name in _LOCOMOTION_TERMS:
    r.pop(name, None)

  # Named gait_tracking so the notebook's weight dict reaches it. The trot's
  # strict std throughout, the jump included: the loose (gallop) std let the
  # policy swap the reference's four-legs-together push for a rear-only kick
  # at almost no tracking cost.
  r["gait_tracking"] = RewardTermCfg(
    func=mdp.jump_tracking,
    weight=0.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "std": GAIT_STD,
      "std_hold": GAIT_STD,
      "loose_from_s": JUMP_ONLY_HOLD_S,
      "loose_until_s": JUMP_ONLY_LOOSE_UNTIL_S,
      "phase_lead_steps": GAIT_REWARD_PHASE_LEAD_STEPS,
      **ref,
    },
  )
  r["jump_up_velocity"] = RewardTermCfg(
    func=mdp.jump_up_velocity,
    weight=0.0,
    params={
      "gate_before_s": JUMP_ONLY_HOLD_S,
      "gate_until_s": JUMP_ONLY_HOLD_S + JUMP_ONLY_TOUCHDOWN_S,
    },
  )
  r["jump_airborne_height"] = RewardTermCfg(
    func=mdp.jump_airborne_height,
    weight=0.0,
    params={
      "stand_height": JUMP_ONLY_STAND_HEIGHT,
      "gate_before_s": JUMP_ONLY_HOLD_S,
      "height_cap": JUMP_ONLY_HEIGHT_CAP,
    },
  )
  r["jump_hold_descent"] = RewardTermCfg(
    func=mdp.jump_hold_descent,
    weight=0.0,
    params={"gate_after_s": JUMP_ONLY_HOLD_S},
  )
  r["torso_clearance"] = RewardTermCfg(
    func=mdp.torso_clearance,
    weight=0.0,
    params={"min_clearance": JUMP_ONLY_TORSO_CLEARANCE},
  )
  return cfg
