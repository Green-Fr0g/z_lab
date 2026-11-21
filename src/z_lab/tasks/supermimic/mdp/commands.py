"""Reference clips and episode sampling for SuperMimic."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

from z_lab.tasks.mimic.mdp.commands import MotionCommand as MimicMotionCommand
from z_lab.tasks.mimic.mdp.commands import MotionCommandCfg as MimicMotionCommandCfg
from z_lab.tasks.mimic.mdp.commands import MotionLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLibrary(MotionLoader):
    """Concatenated z_lab clips in a common joint and tracked-link order."""

    def __init__(
        self,
        files: Sequence[str],
        body_names: Sequence[str],
        root_name: str,
        step_dt: float,
        device: str,
    ) -> None:
        if not files:
            raise ValueError("At least one SuperMimic motion file is required.")
        clips = [MotionLoader(path, body_names=body_names, device=device) for path in files]
        self.fps = clips[0].fps
        self.root_link_name = root_name
        self.joint_names = clips[0].joint_names
        self.motion_body_names = list(body_names)
        self.motion_names = [Path(path).stem for path in files]
        self._body_indexes = slice(None)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError(f"Motion {files[0]!r} needs unique joint names.")

        joint_positions = []
        joint_velocities = []
        for path, clip in zip(files, clips, strict=True):
            if not math.isclose(clip.fps * step_dt, 1.0, rel_tol=1e-5):
                raise ValueError(f"Motion {path!r} is {clip.fps:g} Hz, expected {1 / step_dt:g} Hz.")
            with np.load(path, allow_pickle=True) as data:
                clip_root_name = str(data["root_link_name"].item()) if "root_link_name" in data else None
            if clip_root_name != root_name or clip.time_step_total < 2:
                raise ValueError(f"Motion {path!r} needs root {root_name!r} and at least two frames.")
            if len(clip.joint_names) != len(self.joint_names) or set(clip.joint_names) != set(self.joint_names):
                raise ValueError(f"Motion {path!r} has different joint names from {files[0]!r}.")
            order = [clip.joint_names.index(name) for name in self.joint_names]
            joint_positions.append(clip.joint_pos[:, order])
            joint_velocities.append(clip.joint_vel[:, order])
        self.joint_pos = torch.cat(joint_positions)
        self.joint_vel = torch.cat(joint_velocities)
        self._body_pos_w = torch.cat([clip.body_pos_w for clip in clips])
        self._body_quat_w = torch.cat([clip.body_quat_w for clip in clips])
        self._body_lin_vel_w = torch.cat([clip.body_lin_vel_w for clip in clips])
        self._body_ang_vel_w = torch.cat([clip.body_ang_vel_w for clip in clips])
        self.time_step_total = self.joint_pos.shape[0]
        self.motion_ends = torch.tensor([clip.time_step_total for clip in clips], device=device).cumsum(0)
        self.motion_starts = torch.cat((self.motion_ends.new_zeros(1), self.motion_ends[:-1]))

    def clip_end(self, frames: torch.Tensor) -> torch.Tensor:
        """Return the exclusive end of the clip containing each frame."""
        return self.motion_ends[torch.searchsorted(self.motion_ends, frames.contiguous(), right=True)]

    def body_state(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reference link poses and twists without environment origins."""
        return (
            self._body_pos_w[frames],
            self._body_quat_w[frames],
            self._body_lin_vel_w[frames],
            self._body_ang_vel_w[frames],
        )

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, 0]

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, 0]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, 0]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, 0]


class MotionCommand(MimicMotionCommand):
    """Advance clips and write reference states on episode reset."""

    cfg: MotionCommandCfg
    motion: MotionLibrary

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._sampling_probs = torch.full((self.bin_count,), 1 / self.bin_count, device=self.device)
        if not 0 <= cfg.play_clip < len(self.motion.motion_names):
            raise ValueError(f"Invalid SuperMimic clip index {cfg.play_clip}.")
        clip_length = int(self.motion.motion_ends[cfg.play_clip] - self.motion.motion_starts[cfg.play_clip])
        if not 0 <= cfg.play_frame < clip_length - 1:
            raise ValueError(f"SuperMimic start frame must be in [0, {clip_length - 2}].")

    def _load_motion(self) -> MotionLibrary:
        if self.cfg.body_names[0] != self.cfg.anchor_body_name:
            raise ValueError("The first tracked link must be the floating root and motion anchor.")
        files = self.cfg.motion_file
        return MotionLibrary(
            [files] if isinstance(files, str) else files,
            self.cfg.body_names,
            self.cfg.anchor_body_name,
            self._env.step_dt,
            self.device,
        )

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w.torch[:, self.body_indexes]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.root_link_lin_vel_w.torch

    def set_sampling_distribution(self, probs: torch.Tensor) -> None:
        """Set external RSI bin weights in the concatenated frame timeline."""
        probs = probs.to(self._sampling_probs)
        if probs.shape != self._sampling_probs.shape:
            raise ValueError(f"Expected {self.bin_count} sampling weights.")
        if not torch.isfinite(probs).all() or (probs < 0).any() or probs.sum() <= 0:
            raise ValueError("Sampling weights must be finite, nonnegative, and have a positive sum.")
        self._sampling_probs.copy_(probs / probs.sum())

    def _adaptive_sampling_probs(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Record episode failures and return the distribution for the next reset."""
        failed = self._env.termination_manager.terminated[env_ids] & (self._env.episode_length_buf[env_ids] > 0)
        frame_bins = self.time_steps[env_ids] * self.bin_count // self.motion.time_step_total
        self._current_bin_failed.scatter_add_(0, frame_bins, failed.to(self._current_bin_failed.dtype))

        failure_counts = self.bin_failed_count
        if self.cfg.adaptive_clip > 0.0:
            failure_counts = failure_counts.clamp_max(self.cfg.adaptive_clip * failure_counts.mean())
        weights = failure_counts + self.cfg.adaptive_uniform_ratio / self.bin_count
        if self.cfg.adaptive_kernel_size > 1:
            weights = torch.nn.functional.pad(
                weights.view(1, 1, -1), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate"
            )
            weights = torch.nn.functional.conv1d(weights, self.kernel.view(1, 1, -1)).flatten()
        total = weights.sum()
        probs = torch.where(total > 0, weights / total, 1.0 / self.bin_count)

        top_prob, top_bin = probs.max(dim=0)
        self.metrics["sampling_entropy"][:] = (
            torch.special.entr(probs).sum() / math.log(self.bin_count) if self.bin_count > 1 else 1.0
        )
        self.metrics["sampling_top1_prob"][:] = top_prob
        self.metrics["sampling_top1_bin"][:] = top_bin / self.bin_count
        return probs

    def _resample_command(self, env_ids: Sequence[int] | slice) -> None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        count = len(env_ids)
        if not count:
            return

        if self.cfg.sampling_mode == "start":
            start = self.motion.motion_starts[self.cfg.play_clip] + self.cfg.play_frame
            frames = start.expand(count)
        elif self.cfg.sampling_mode in ("adaptive", "external"):
            probs = (
                self._adaptive_sampling_probs(env_ids) if self.cfg.sampling_mode == "adaptive" else self._sampling_probs
            )
            bins = torch.multinomial(probs, count, replacement=True)
            fractions = (bins + torch.rand(count, device=self.device)) / self.bin_count
            frames = (fractions * self.motion.time_step_total).long().clamp_max(self.motion.time_step_total - 1)
            frames = torch.minimum(frames, self.motion.clip_end(frames) - 2)
        else:
            # One slot per valid start frame, omitting every clip's last frame.
            valid_ends = self.motion.motion_ends - torch.arange(
                1, len(self.motion.motion_names) + 1, device=self.device
            )
            slots = torch.randint(
                self.motion.time_step_total - len(self.motion.motion_names), (count,), device=self.device
            )
            frames = slots + torch.searchsorted(valid_ends, slots, right=True)
        self.time_steps[env_ids] = frames

        pose_delta = self._sample_root_delta(self.cfg.pose_range, count)
        twist_delta = self._sample_root_delta(self.cfg.velocity_range, count)
        pos = self.motion.root_pos_w[frames] + self._env.scene.env_origins[env_ids] + pose_delta[:, :3]
        quat = quat_mul(quat_from_euler_xyz(*pose_delta[:, 3:].unbind(-1)), self.motion.root_quat_w[frames])
        twist = torch.cat((self.motion.root_lin_vel_w[frames], self.motion.root_ang_vel_w[frames]), dim=-1)
        joint_pos = self.motion.joint_pos[frames]
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, self.device)
        lo, hi = self.robot.data.soft_joint_pos_limits.torch[env_ids].unbind(dim=-1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=twist + twist_delta, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos.clamp(lo, hi), env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=self.motion.joint_vel[frames], env_ids=env_ids)
        # Explicit env.reset() does not run the command manager's compute step.
        reference_pos, reference_quat = self.aligned_body_poses()
        self.body_pos_relative_w[env_ids] = reference_pos[env_ids]
        self.body_quat_relative_w[env_ids] = reference_quat[env_ids]

    def _sample_root_delta(self, ranges: dict[str, tuple[float, float]], count: int) -> torch.Tensor:
        axes = ("x", "y", "z", "roll", "pitch", "yaw")
        bounds = torch.tensor([ranges.get(axis, (0.0, 0.0)) for axis in axes], device=self.device)
        return sample_uniform(bounds[:, 0], bounds[:, 1], (count, 6), self.device)

    def _update_command(self) -> None:
        # Same-step reset runs before the command manager. Keep the sampled frame for those rows.
        next_frames = self.time_steps + (self._env.episode_length_buf > 0)
        self.time_steps.copy_(torch.minimum(next_frames, self.motion.clip_end(self.time_steps) - 1))
        self.body_pos_relative_w, self.body_quat_relative_w = self.aligned_body_poses()
        if self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count.lerp_(self._current_bin_failed, self.cfg.adaptive_alpha)
            self._current_bin_failed.zero_()


@configclass
class MotionCommandCfg(MimicMotionCommandCfg):
    """Configure reference clips, episode sampling, and playback."""

    class_type: type = MotionCommand
    motion_file: str | list[str]
    sampling_mode: Literal["start", "uniform", "external", "adaptive"] = "uniform"
    adaptive_clip: float = 100.0
    """Cap per-bin failure counts at this multiple of their mean. Use zero to disable the cap."""
    play_clip: int = 0
    play_frame: int = 0
