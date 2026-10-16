"""Online AMP reference motion management for the canonical z_lab NPZ format."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isaaclab.assets import Articulation


def _as_str_list(values) -> list[str]:
    return [str(name) for name in np.asarray(values).tolist()]


def _index_by_names(available: Sequence[str], requested: Sequence[str], kind: str) -> list[int]:
    lookup = {name: index for index, name in enumerate(available)}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise ValueError(f"AMP motion {kind} names {missing} were not found. Available {kind} names: {list(available)}")
    return [lookup[name] for name in requested]


def _value(value):
    return value.torch if hasattr(value, "torch") else value


@dataclass
class MotionClip:
    """A canonical motion clip kept on the simulation device."""

    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    dof_names: tuple[str, ...]

    @property
    def num_frames(self) -> int:
        return self.body_pos_w.shape[0]


class MotionResetManager:
    """Load motion clips and maintain one online reference phase per environment."""

    _instance: MotionResetManager | None = None

    def __init__(self) -> None:
        self._clips: dict[str, list[MotionClip]] = {}
        self._recovery_clips: dict[str, list[MotionClip]] = {}
        self._cache_key = ""
        self._body_names: tuple[str, ...] = ()
        self._clip_ids: torch.Tensor | None = None
        self._frame_ids: torch.Tensor | None = None
        self._recovery_elapsed_steps: torch.Tensor | None = None
        self._recovery_max_steps = 0
        self._recovery_active: torch.Tensor | None = None
        self._recovery_succeeded: torch.Tensor | None = None
        self._recovery_failed: torch.Tensor | None = None
        self._recovery_selected: torch.Tensor | None = None
        self._recovery_success_counter: torch.Tensor | None = None
        self._reference: dict[str, torch.Tensor] = {}
        self._env_origins: torch.Tensor | None = None
        self._last_recovery_success_rate = 0.0
        self._last_recovery_active_ratio = 0.0

    def _ensure_state(self, num_envs: int, device: torch.device) -> None:
        """Create online phase and reference buffers before the first reset."""
        if (
            self._clip_ids is not None
            and self._clip_ids.shape[0] == num_envs
            and self._reference.get("body_pos_w", torch.empty(0)).shape[:2] == (num_envs, len(self._body_names))
        ):
            return
        self._clip_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._frame_ids = torch.zeros_like(self._clip_ids)
        self._recovery_elapsed_steps = torch.zeros_like(self._clip_ids)
        self._recovery_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._recovery_succeeded = torch.zeros_like(self._recovery_active)
        self._recovery_failed = torch.zeros_like(self._recovery_active)
        self._recovery_selected = torch.zeros_like(self._recovery_active)
        self._recovery_success_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        shape = (num_envs, len(self._body_names))
        self._reference = {
            "body_pos_w": torch.zeros((*shape, 3), dtype=torch.float32, device=device),
            "body_quat_w": torch.zeros((*shape, 4), dtype=torch.float32, device=device),
            "body_lin_vel_w": torch.zeros((*shape, 3), dtype=torch.float32, device=device),
            "body_ang_vel_w": torch.zeros((*shape, 3), dtype=torch.float32, device=device),
        }
        self._reference["body_quat_w"][..., 0] = 1.0

    @classmethod
    def get(cls) -> MotionResetManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def recovery_active(self) -> torch.Tensor:
        if self._recovery_active is None:
            raise RuntimeError("AMP motion reference has not been reset yet.")
        return self._recovery_active

    @property
    def recovery_succeeded(self) -> torch.Tensor:
        if self._recovery_succeeded is None:
            raise RuntimeError("AMP motion reference has not been reset yet.")
        return self._recovery_succeeded

    @property
    def recovery_failed(self) -> torch.Tensor:
        if self._recovery_failed is None:
            raise RuntimeError("AMP motion reference has not been reset yet.")
        return self._recovery_failed

    @property
    def recovery_selected(self) -> torch.Tensor:
        if self._recovery_selected is None:
            raise RuntimeError("AMP motion reference has not been reset yet.")
        return self._recovery_selected

    @property
    def last_recovery_metrics(self) -> tuple[float, float]:
        """Return the success rate and recovery fraction from the previous reset batch."""
        return self._last_recovery_success_rate, self._last_recovery_active_ratio

    @property
    def body_names(self) -> tuple[str, ...]:
        return self._body_names

    def init(
        self,
        motion_dir: str,
        device: str | torch.device,
        root_name: str,
        all_body_names: tuple[str, ...],
        recovery_motion_dir: str | None = None,
        recovery_name_patterns: Sequence[str] = ("fallAndGetUp",),
        num_envs: int | None = None,
    ) -> None:
        """Load normal and recovery clips without building discriminator transitions."""
        del root_name
        motion_path = str(Path(motion_dir).expanduser().resolve())
        recovery_path = str(Path(recovery_motion_dir).expanduser().resolve()) if recovery_motion_dir else motion_path
        body_names = tuple(all_body_names)
        cache_key = f"{motion_path}:{recovery_path}:{body_names}:{tuple(recovery_name_patterns)}"
        if cache_key == self._cache_key:
            if num_envs is not None:
                self._ensure_state(num_envs, torch.device(device))
            return

        normal_files = self._collect_motion_files(motion_path)
        recovery_files = self._collect_motion_files(recovery_path)
        if recovery_path == motion_path:
            recovery_files = [
                path for path in recovery_files if any(pattern in path.stem for pattern in recovery_name_patterns)
            ]
            normal_files = [
                path for path in normal_files if not any(pattern in path.stem for pattern in recovery_name_patterns)
            ]

        device_obj = torch.device(device)
        self._clips[cache_key] = [self._load_clip(path, body_names, device_obj) for path in normal_files]
        self._recovery_clips[cache_key] = [self._load_clip(path, body_names, device_obj) for path in recovery_files]
        if not self._clips[cache_key]:
            raise FileNotFoundError(f"No normal AMP motion files found in: {motion_path}")
        self._cache_key = cache_key
        self._body_names = body_names
        if num_envs is not None:
            self._ensure_state(num_envs, device_obj)

    def reset(
        self,
        env,
        env_ids: torch.Tensor | None,
        motion_dir: str,
        root_name: str,
        all_body_names: tuple[str, ...],
        asset_cfg,
        recovery_motion_dir: str | None = None,
        recovery_env_ratio: float = 0.0,
        recovery_name_patterns: Sequence[str] = ("fallAndGetUp",),
        recovery_max_steps: int = 250,
    ) -> None:
        """Sample reference frames, write the robot state, and initialize online phase buffers."""
        self.init(
            motion_dir,
            env.device,
            root_name,
            all_body_names,
            recovery_motion_dir=recovery_motion_dir,
            recovery_name_patterns=recovery_name_patterns,
            num_envs=env.num_envs,
        )
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        if len(env_ids) == 0:
            return

        if self._recovery_selected is not None:
            previous = self._recovery_selected[env_ids]
            denominator = previous.sum().clamp_min(1)
            self._last_recovery_success_rate = (
                (previous & self._recovery_succeeded[env_ids]).sum() / denominator
            ).item()
            self._last_recovery_active_ratio = previous.float().mean().item()

        self._ensure_state(env.num_envs, env.device)
        self._env_origins = env.scene.env_origins

        key = self._cache_key
        normal_clips = self._clips[key]
        recovery_clips = self._recovery_clips[key]
        use_recovery = torch.rand(len(env_ids), device=env.device) < recovery_env_ratio
        if not recovery_clips:
            use_recovery.zero_()
        self._recovery_active[env_ids] = use_recovery
        self._recovery_elapsed_steps[env_ids] = 0
        self._recovery_max_steps = int(recovery_max_steps)
        self._recovery_selected[env_ids] = use_recovery
        self._recovery_success_counter[env_ids] = 0
        self._recovery_succeeded[env_ids] = False
        self._recovery_failed[env_ids] = False

        selected_clip_ids = torch.zeros(len(env_ids), dtype=torch.long, device=env.device)
        selected_frame_ids = torch.zeros_like(selected_clip_ids)
        asset: Articulation = env.scene[asset_cfg.name]
        asset_joint_names = tuple(asset.joint_names)
        root_index = self._body_names.index(root_name)
        root_pos = torch.empty((len(env_ids), 3), device=env.device)
        root_quat = torch.empty((len(env_ids), 4), device=env.device)
        root_lin_vel = torch.empty((len(env_ids), 3), device=env.device)
        root_ang_vel = torch.empty((len(env_ids), 3), device=env.device)
        joint_pos = torch.empty((len(env_ids), len(asset_joint_names)), device=env.device)
        joint_vel = torch.empty_like(joint_pos)

        for is_recovery in (False, True):
            selected = use_recovery == is_recovery
            if not selected.any():
                continue
            clips = recovery_clips if is_recovery else normal_clips
            selected_positions = selected.nonzero(as_tuple=False).squeeze(-1)
            # Sample the frame independently for every environment, even when
            # several environments selected the same motion clip.
            clip_ids, frame_ids = self._sample_clip_frame_ids(clips, selected_positions.numel(), env.device)
            selected_clip_ids[selected_positions] = clip_ids
            selected_frame_ids[selected_positions] = frame_ids
            for clip_id in clip_ids.unique().tolist():
                clip_selected = clip_ids == clip_id
                target = selected_positions[clip_selected]
                clip = clips[clip_id]
                ids = frame_ids[clip_selected]
                root_pos[target] = clip.body_pos_w[ids, root_index]
                root_quat[target] = clip.body_quat_w[ids, root_index]
                root_lin_vel[target] = clip.body_lin_vel_w[ids, root_index]
                root_ang_vel[target] = clip.body_ang_vel_w[ids, root_index]
                joint_indexes = _index_by_names(clip.dof_names, asset_joint_names, "dof")
                joint_pos[target] = clip.dof_pos[ids][:, joint_indexes]
                joint_vel[target] = clip.dof_vel[ids][:, joint_indexes]
                env_target = env_ids[target]
                self._write_reference(env_target, clip, ids)

        self._clip_ids[env_ids] = selected_clip_ids
        self._frame_ids[env_ids] = selected_frame_ids
        root_pos = root_pos + env.scene.env_origins[env_ids]
        joint_ids = asset_cfg.joint_ids
        joint_pos = joint_pos[:, joint_ids]
        joint_vel = joint_vel[:, joint_ids]
        limits = _value(asset.data.soft_joint_pos_limits)[env_ids][:, joint_ids]
        joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
        asset.write_root_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.cat((root_lin_vel, root_ang_vel), dim=-1), env_ids=env_ids)
        asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)

    def advance(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Advance references and return environments that changed motion clips."""
        if self._frame_ids is None:
            return torch.empty(0, dtype=torch.long)
        if env_ids is None:
            env_ids = torch.arange(self._frame_ids.shape[0], device=self._frame_ids.device)
        key = self._cache_key
        normal_clips = self._clips[key]
        recovery_clips = self._recovery_clips[key]
        changed_env_ids: list[torch.Tensor] = []
        for is_recovery in (False, True):
            selected = self._recovery_active[env_ids] == is_recovery
            if not selected.any():
                continue
            target_ids = env_ids[selected]
            clips = recovery_clips if is_recovery else normal_clips
            if is_recovery and self._recovery_elapsed_steps is not None:
                self._recovery_elapsed_steps[target_ids] += 1
            clip_ids = self._clip_ids[target_ids]
            frame_ids = self._frame_ids[target_ids] + 1
            for clip_id in clip_ids.unique().tolist():
                clip_selected = clip_ids == clip_id
                target = target_ids[clip_selected]
                ids = frame_ids[clip_selected]
                clip = clips[clip_id]
                # When a clip ends, select a fresh clip and phase instead of
                # wrapping the same motion back to frame zero.
                at_end = ids >= clip.num_frames
                if at_end.any():
                    changed_env_ids.append(self._resample_motion(target[at_end], clips))
                target = target[~at_end]
                ids = ids[~at_end]
                self._frame_ids[target] = ids
                if target.numel() > 0:
                    self._write_reference(target, clip, ids)
        if not changed_env_ids:
            return env_ids.new_empty((0,))
        return torch.cat(changed_env_ids).unique()

    def reference_body_data(self, anchor_name: str, body_names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """Return online reference body data selected by semantic names."""
        if not self._reference:
            raise RuntimeError("AMP reference observations were requested before motion reset.")
        anchor_id = self._body_names.index(anchor_name)
        body_ids = _index_by_names(self._body_names, tuple(body_names), "reference body")
        return (
            self._reference["body_pos_w"][:, anchor_id],
            self._reference["body_quat_w"][:, anchor_id],
            self._reference["body_pos_w"][:, body_ids],
            self._reference["body_quat_w"][:, body_ids],
            self._reference["body_lin_vel_w"][:, body_ids],
            self._reference["body_ang_vel_w"][:, body_ids],
        )

    def mark_success(self, env_ids: torch.Tensor) -> torch.Tensor:
        self._recovery_succeeded[env_ids] = True
        self._recovery_active[env_ids] = False
        return self._resample_motion(env_ids, self._clips[self._cache_key])

    def _resample_motion(self, env_ids: torch.Tensor, clips: list[MotionClip]) -> torch.Tensor:
        """Assign fresh clip and frame ids and update the online reference."""
        if env_ids.numel() == 0:
            return env_ids
        clip_ids, frame_ids = self._sample_clip_frame_ids(clips, env_ids.numel(), env_ids.device)
        self._clip_ids[env_ids] = clip_ids
        self._frame_ids[env_ids] = frame_ids
        for clip_id in clip_ids.unique().tolist():
            selected = clip_ids == clip_id
            target = env_ids[selected]
            clip = clips[clip_id]
            ids = frame_ids[selected]
            self._write_reference(target, clip, ids)
        return env_ids

    @staticmethod
    def _sample_clip_frame_ids(
        clips: list[MotionClip], num_envs: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample independent clip and frame ids for a batch of environments."""
        clip_ids = torch.randint(len(clips), (num_envs,), device=device)
        frame_ids = torch.stack(
            [torch.randint(clips[int(clip_id)].num_frames, (), device=device) for clip_id in clip_ids.tolist()]
        )
        return clip_ids, frame_ids

    def _write_reference(self, env_ids: torch.Tensor, clip: MotionClip, frame_ids: torch.Tensor) -> None:
        """Write one clip's body reference data for selected environments."""
        self._reference["body_pos_w"][env_ids] = clip.body_pos_w[frame_ids] + self._env_origins[env_ids, None, :]
        self._reference["body_quat_w"][env_ids] = clip.body_quat_w[frame_ids]
        self._reference["body_lin_vel_w"][env_ids] = clip.body_lin_vel_w[frame_ids]
        self._reference["body_ang_vel_w"][env_ids] = clip.body_ang_vel_w[frame_ids]

    def mark_failure(self, env_ids: torch.Tensor) -> None:
        self._recovery_failed[env_ids] = True
        self._recovery_active[env_ids] = False

    def update_recovery_status(
        self,
        env,
        success_height: float,
        success_angle: float,
        success_hold_steps: int,
    ) -> torch.Tensor:
        """Update recovery success state from the simulated root pose."""
        if self._recovery_active is None or not self._recovery_active.any():
            device = self._recovery_active.device if self._recovery_active is not None else env.device
            return torch.empty(0, dtype=torch.long, device=device)
        asset: Articulation = env.scene["robot"]
        active_ids = self._recovery_active.nonzero(as_tuple=False).squeeze(-1)
        root_height = _value(asset.data.root_pos_w)[active_ids, 2] - env.scene.env_origins[active_ids, 2]
        projected_gravity = _value(asset.data.projected_gravity_b)[active_ids]
        tilt = torch.acos((-projected_gravity[:, 2]).clamp(-1.0, 1.0))
        upright = (root_height >= success_height) & (tilt <= success_angle)
        counters = self._recovery_success_counter[active_ids]
        counters = torch.where(upright, counters + 1, torch.zeros_like(counters))
        self._recovery_success_counter[active_ids] = counters
        done = counters >= max(int(success_hold_steps), 1)
        if done.any():
            return self.mark_success(active_ids[done])
        return active_ids.new_empty((0,))

    def assist_scale(
        self,
        env,
        force_start: float | None = None,
        force_end: float | None = None,
        anneal_iterations: int | None = None,
    ) -> float:
        iteration = float(getattr(env, "_amp_training_iteration", 0))
        if anneal_iterations is None:
            anneal_iterations = getattr(env.cfg, "assist_anneal_iterations", 50000)
        if force_start is None:
            force_start = getattr(env.cfg, "assist_force_start", 0.0)
        if force_end is None:
            force_end = getattr(env.cfg, "assist_force_end", 0.0)
        anneal_iterations = max(float(anneal_iterations), 1.0)
        start = float(force_start)
        end = float(force_end)
        alpha = min(max(iteration / anneal_iterations, 0.0), 1.0)
        return start + (end - start) * alpha

    def recovery_clip_finished(self) -> torch.Tensor:
        """Return recovery environments that exceeded their allowed duration."""
        if self._recovery_active is None or self._frame_ids is None:
            return torch.zeros(0, dtype=torch.bool)
        finished = torch.zeros_like(self._recovery_active)
        recovery_ids = self._recovery_active.nonzero(as_tuple=False).squeeze(-1)
        if recovery_ids.numel() == 0:
            return finished
        if self._recovery_max_steps > 0 and self._recovery_elapsed_steps is not None:
            finished[recovery_ids] |= self._recovery_elapsed_steps[recovery_ids] >= self._recovery_max_steps
        return finished

    @staticmethod
    def _load_clip(path: Path, all_body_names: Sequence[str], device: torch.device) -> MotionClip:
        data = np.load(path)
        required = (
            "body_positions",
            "body_rotations",
            "body_linear_velocities",
            "body_angular_velocities",
            "dof_positions",
            "dof_velocities",
            "body_names",
            "dof_names",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"AMP motion file '{path}' is missing keys: {missing}")
        motion_body_names = _as_str_list(data["body_names"])
        body_indexes = _index_by_names(motion_body_names, all_body_names, "body")
        body_pos_w = torch.as_tensor(data["body_positions"], dtype=torch.float32, device=device)[:, body_indexes]
        body_quat_w = torch.as_tensor(data["body_rotations"], dtype=torch.float32, device=device)[:, body_indexes]
        body_lin_vel_w = torch.as_tensor(data["body_linear_velocities"], dtype=torch.float32, device=device)[
            :, body_indexes
        ]
        body_ang_vel_w = torch.as_tensor(data["body_angular_velocities"], dtype=torch.float32, device=device)[
            :, body_indexes
        ]
        dof_pos = torch.as_tensor(data["dof_positions"], dtype=torch.float32, device=device)
        dof_vel = torch.as_tensor(data["dof_velocities"], dtype=torch.float32, device=device)
        dof_names = tuple(_as_str_list(data["dof_names"]))
        if dof_pos.ndim != 2 or dof_vel.shape != dof_pos.shape or len(dof_names) != dof_pos.shape[1]:
            raise ValueError(f"AMP motion file '{path}' has inconsistent dof_names/dof_positions/dof_velocities.")
        return MotionClip(body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, dof_pos, dof_vel, dof_names)

    @staticmethod
    def _collect_motion_files(path_string: str) -> list[Path]:
        path = Path(path_string)
        if path.is_file() and path.suffix == ".npz":
            return [path]
        files = sorted(path.rglob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No AMP motion .npz files found in: {path}")
        return files
