"""Play a SuperMimic actor from a local checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import warp as wp
import yaml
from cli_args import add_common_args, checkpoint_path, parse_task_configs
from isaaclab_visualizers.newton.newton_visualization_markers import NewtonVisualizationMarkers
from newton.viewer import ViewerBase
from pxr import Usd, UsdGeom, UsdPhysics
from supermimic.algorithms.models import MultiEncoderAE, TaskSpec
from supermimic.algorithms.representations import extract_actor_goal
from supermimic.envs.isaaclab_adapter import IsaacLabSuperMimicAdapter
from supermimic.envs.world_model_env import WorldModelRollout
from supermimic.kinematics import ForwardKinematics, load_robot_model
from supermimic.runners.exporter import export_actor
from supermimic.runners.runner import build_world_model_from_state
from supermimic.tasks import VelocityTaskCfg, advance_proprio_history, build_actor_obs_td
from supermimic.utils.math import matrix_to_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul

from isaaclab.app import launch_simulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import PreviewSurfaceCfg, UsdFileCfg

from z_lab.tasks.supermimic.tracking_env_cfg import SuperMimicGenCoordEnvCfg


def load_checkpoint(path: Path, representation: str, task_encoder: str) -> dict:
    """Load and validate a local proprioceptive playback checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("actor_obs_mode") != "proprio":
        raise ValueError("Playback requires a checkpoint trained with proprioception. Retrain the rep_state actor.")
    if checkpoint["representation"] != representation:
        raise ValueError("Checkpoint representation must match the playback agent")
    if task_encoder not in checkpoint["task_names"]:
        raise ValueError(f"checkpoint has no {task_encoder!r} encoder; train with that task enabled")
    return checkpoint


def load_training_snapshot(path: Path) -> dict:
    """Read the saved environment configuration as plain data."""
    if not path.is_file():
        return {}

    class SnapshotLoader(yaml.SafeLoader):
        pass

    for tag in ("python/tuple", "python/object/apply:builtins.slice"):
        SnapshotLoader.add_constructor(f"tag:yaml.org,2002:{tag}", SnapshotLoader.construct_sequence)
    with path.open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=SnapshotLoader) or {}


def load_training_motion_file(env_cfg: SuperMimicGenCoordEnvCfg, motion_file: Path | None, snapshot: dict) -> None:
    """Use the training motion library unless the caller supplied a file."""
    training_motion = snapshot.get("commands", {}).get("motion", {}).get("motion_file")
    if motion_file is None and training_motion is not None:
        env_cfg.commands.motion.motion_file = training_motion


def build_actor(adapter: IsaacLabSuperMimicAdapter, checkpoint: dict, task_encoder: str) -> MultiEncoderAE:
    """Restore the actor and route the entire batch through the selected encoder."""
    observations = adapter.observations()
    obs_dim = adapter.validate_proprio_layout()
    adapter.validate_full_state_layout(checkpoint["representation"], observations["full_state"].shape[-1])
    if list(adapter.action_term._joint_names) != checkpoint["joint_names"]:
        raise ValueError("Checkpoint joints must match the playback task's action order")
    if list(adapter.motion_command.cfg.body_names) != checkpoint["body_names"]:
        raise ValueError("Checkpoint links must match the playback task's tracking order")
    saved_config = checkpoint["config"]
    frames, stride = adapter.motion_goal_window()
    if (frames, stride) != (saved_config["num_goal_frames"], saved_config["goal_frame_stride"]):
        raise ValueError("Checkpoint Motion window must match the playback task")
    goal_dim = extract_actor_goal(observations["goal"], frames).shape[-1]
    goal_dims = {"motion": goal_dim, "velocity": 3}
    task_names = checkpoint["task_names"]
    actor = MultiEncoderAE(
        obs_dim=obs_dim,
        action_dim=adapter.num_actions,
        task_specs=[TaskSpec(name, goal_dims[name]) for name in task_names],
        slot_mix={task_encoder: 1.0},
        latent_dim=saved_config["latent_dim"],
        hidden_dims=tuple(saved_config["actor_hidden_dims"]),
        history_length=saved_config["history_length"],
    ).to(adapter.device)
    actor.load_state_dict(checkpoint["actor"])
    return actor.eval()


def sample_velocity_commands(
    commands: torch.Tensor,
    cfg: VelocityTaskCfg,
    fixed_command: list[float] | None,
    env_ids: torch.Tensor,
) -> None:
    """Replace commands on reset rows, with a supplied command or independent uniform samples."""
    if fixed_command is not None:
        commands[env_ids] = commands.new_tensor(fixed_command)
        return
    bounds = commands.new_tensor((cfg.lin_vel_x_range, cfg.lin_vel_y_range, cfg.ang_vel_z_range))
    lower, upper = bounds.unbind(-1)
    samples = torch.rand((env_ids.numel(), 3), dtype=commands.dtype, device=commands.device)
    commands[env_ids] = lower + (upper - lower) * samples


class _ShadowViewer:
    """Forward marker draws to a Newton viewer with a shared instance opacity."""

    def __init__(self, viewer: ViewerBase, opacity: float) -> None:
        self._viewer = viewer
        self._opacity = opacity
        self._opacities: dict[int, wp.array] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._viewer, name)

    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms: wp.array | None,
        scales: wp.array | None,
        colors: wp.array | None,
        materials: wp.array | None,
        hidden: bool = False,
    ) -> None:
        opacities = None
        if xforms is not None and not hidden:
            count = len(xforms)
            if count not in self._opacities:
                self._opacities[count] = wp.full(count, self._opacity, dtype=wp.float32, device=self._viewer.device)
            opacities = self._opacities[count]
        self._viewer.log_instances(name, mesh, xforms, scales, colors, materials, hidden=hidden, opacities=opacities)


class _ShadowNewtonMarkers(NewtonVisualizationMarkers):
    """Use Isaac Lab's mesh batching with explicit Newton instance opacity."""

    def __init__(self, cfg: VisualizationMarkersCfg, visible: bool, opacity: float) -> None:
        self._opacity = opacity
        self._shadow_viewers: dict[ViewerBase, _ShadowViewer] = {}
        super().__init__(cfg, visible=visible)

    def render(
        self, viewer: ViewerBase, visible_env_ids: list[int] | None, num_envs: int, render_id: str | None = None
    ) -> None:
        if viewer not in self._shadow_viewers:
            self._shadow_viewers[viewer] = _ShadowViewer(viewer, self._opacity)
        super().render(cast(ViewerBase, self._shadow_viewers[viewer]), visible_env_ids, num_envs, render_id)


class _ShadowMarkers(VisualizationMarkers):
    """Apply the same opacity in USD materials and Newton instance draws."""

    def __init__(self, cfg: VisualizationMarkersCfg, opacity: float) -> None:
        self._opacity = opacity
        super().__init__(cfg)

    def _ensure_newton_backend(self) -> None:
        if not any(isinstance(backend, _ShadowNewtonMarkers) for backend in self._backends):
            self._backends.append(_ShadowNewtonMarkers(self.cfg, self._is_visible, self._opacity))


def _create_shadow_markers(
    name: str, meshes: dict[str, Path], color: tuple[float, float, float]
) -> VisualizationMarkers:
    opacity = 0.5
    cfg = VisualizationMarkersCfg(
        prim_path=f"/Visuals/SuperMimic/{name}",
        markers={
            link: UsdFileCfg(
                usd_path=str(path), visual_material=PreviewSurfaceCfg(diffuse_color=color, opacity=opacity)
            )
            for link, path in meshes.items()
        },
    )
    return _ShadowMarkers(cfg, opacity)


def _export_link_meshes(stage: Usd.Stage, directory: Path) -> dict[str, Path]:
    transforms = UsdGeom.XformCache()
    metres = UsdGeom.GetStageMetersPerUnit(stage)
    meshes_by_link: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == UsdGeom.Tokens.invisible or mesh.ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        body = prim.GetParent()
        collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            collision |= body.HasAPI(UsdPhysics.CollisionAPI)
            body = body.GetParent()
        if not body or collision:
            continue
        world_to_body = transforms.GetLocalToWorldTransform(body).GetInverse()
        transform = np.asarray(transforms.GetLocalToWorldTransform(prim) * world_to_body)
        points = np.asarray(mesh.GetPointsAttr().Get())
        points = (points @ transform[:3, :3] + transform[3, :3]) * metres
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        mirrored = np.linalg.det(transform[:3, :3]) < 0
        if mirrored != (mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded):
            indices = np.concatenate([face[::-1] for face in np.split(indices, counts.cumsum()[:-1])])
        meshes_by_link.setdefault(body.GetName(), []).append((points, counts, indices))

    paths = {}
    for link, parts in meshes_by_link.items():
        points, counts, indices = zip(*parts, strict=True)
        offsets = np.cumsum([0, *(len(part) for part in points[:-1])])
        path = directory / f"{link}.usdc"
        link_stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(link_stage, 1.0)
        mesh = UsdGeom.Mesh.Define(link_stage, "/Mesh")
        link_stage.SetDefaultPrim(mesh.GetPrim())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreatePointsAttr(np.concatenate(points).astype(np.float32))
        mesh.CreateFaceVertexCountsAttr(np.concatenate(counts).astype(np.int32))
        mesh.CreateFaceVertexIndicesAttr(
            np.concatenate([part + offset for part, offset in zip(indices, offsets, strict=True)]).astype(np.int32)
        )
        link_stage.GetRootLayer().Save()
        paths[link] = path
    if not paths:
        raise ValueError(f"No visual link meshes found in {stage.GetRootLayer().identifier!r}")
    return paths


class PlaybackShadows:
    """Update visual-only robot link meshes from reference and predicted poses."""

    def __init__(
        self, adapter: IsaacLabSuperMimicAdapter, directory: Path, *, show_reference: bool, show_world_model: bool
    ) -> None:
        self.adapter = adapter
        scene = adapter.env.scene
        command = adapter.motion_command
        robot = command.robot
        stage = Usd.Stage.Open(robot.cfg.spawn.usd_path)
        meshes = _export_link_meshes(stage, directory)
        self.reference = (
            _create_shadow_markers("reference_shadow", meshes, (0.15, 0.75, 1.0)) if show_reference else None
        )
        self.world_model = (
            _create_shadow_markers("world_model_shadow", meshes, (1.0, 0.45, 0.1)) if show_world_model else None
        )
        visual_links = tuple(meshes)
        tracked_links = tuple(command.cfg.body_names)
        fk_links = tuple(dict.fromkeys((*tracked_links, *visual_links)))
        joint_names = robot.joint_names
        model = load_robot_model(stage.GetDefaultPrim(), joint_names, fk_links)
        self.fk = ForwardKinematics(model, dtype=adapter.default_joint_pos().dtype, device=adapter.device)
        self.default_joint_pos = robot.data.default_joint_pos.torch
        self.default_joint_vel = robot.data.default_joint_vel.torch
        self.policy_joint_ids = [joint_names.index(name) for name in adapter.action_term._joint_names]
        motion_names = command.motion.joint_names
        motion_joints = [name for name in joint_names if name in motion_names]
        self.motion_joint_ids = [motion_names.index(name) for name in motion_joints]
        self.motion_robot_joint_ids = [joint_names.index(name) for name in motion_joints]
        self.visual_ids = [fk_links.index(name) for name in visual_links]
        tracked = set(tracked_links)
        self.anchor_ids = [tracked_links.index(model.path(name, tracked)[0]) for name in visual_links]
        self.origins = scene.env_origins[:, None, :]
        self.marker_ids = torch.arange(len(visual_links), device=adapter.device).repeat(adapter.num_envs)
        self.environment_ids = torch.arange(adapter.num_envs, device=adapter.device).repeat_interleave(
            len(visual_links)
        )

    def update_reference_shadow(self) -> None:
        """Draw the current frame of each environment's reference clip."""
        if self.reference is None:
            return
        command = self.adapter.motion_command
        frames = command.time_steps
        motion = command.motion
        joint_pos = self.default_joint_pos.clone()
        joint_vel = self.default_joint_vel.clone()
        joint_pos[:, self.motion_robot_joint_ids] = motion.joint_pos[frames[:, None], self.motion_joint_ids]
        joint_vel[:, self.motion_robot_joint_ids] = motion.joint_vel[frames[:, None], self.motion_joint_ids]
        positions, orientations, _, _ = self.fk(
            motion.root_pos_w[frames],
            motion.root_quat_w[frames],
            joint_pos,
            motion.root_lin_vel_w[frames],
            motion.root_ang_vel_w[frames],
            joint_vel,
        )
        self._draw(self.reference, positions[:, self.visual_ids], orientations[:, self.visual_ids])

    def update_world_model_shadow(self, rollout: WorldModelRollout) -> None:
        """Draw represented link poses and attach any untracked visual links with FK."""
        if self.world_model is None:
            return
        representation = rollout.representation
        positions, rotations, lin_vel, ang_vel = representation.world_frame_bodies(rollout.aux_state, rollout.rep_state)
        orientations = matrix_to_quat(rotations)
        policy_pos, policy_vel = representation.joint_state(rollout.rep_state).chunk(2, dim=-1)
        joint_pos = self.default_joint_pos.clone()
        joint_vel = self.default_joint_vel.clone()
        joint_pos[:, self.policy_joint_ids] = policy_pos
        joint_vel[:, self.policy_joint_ids] = policy_vel
        fk_pos, fk_quat, _, _ = self.fk(
            positions[:, 0], orientations[:, 0], joint_pos, lin_vel[:, 0], ang_vel[:, 0], joint_vel
        )
        anchor_pos, anchor_quat = fk_pos[:, self.anchor_ids], fk_quat[:, self.anchor_ids]
        local_pos = quat_apply_inverse(anchor_quat, fk_pos[:, self.visual_ids] - anchor_pos)
        local_quat = quat_mul(quat_inv(anchor_quat), fk_quat[:, self.visual_ids])
        target_pos, target_quat = positions[:, self.anchor_ids], orientations[:, self.anchor_ids]
        self._draw(self.world_model, target_pos + quat_apply(target_quat, local_pos), quat_mul(target_quat, local_quat))

    def _draw(self, markers: VisualizationMarkers, positions: torch.Tensor, orientations: torch.Tensor) -> None:
        markers.visualize(
            translations=(positions + self.origins).flatten(0, 1),
            orientations=orientations.flatten(0, 1),
            marker_indices=self.marker_ids,
            environment_ids=self.environment_ids,
        )


def reset_playback_state(
    adapter: IsaacLabSuperMimicAdapter,
    observations: dict[str, torch.Tensor],
    rollout: WorldModelRollout | None,
    training_events: dict | None,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Synchronize the indicated model rows with post-reset simulator state."""
    if rollout is not None:
        rollout.reset_dynamics(
            observations["full_state"], adapter.sim_context(training_events), adapter.previous_actions(), env_ids
        )


@torch.no_grad()
def run_playback(
    args: argparse.Namespace,
    adapter: IsaacLabSuperMimicAdapter,
    actor: MultiEncoderAE,
    velocity_cfg: VelocityTaskCfg,
    shadows: PlaybackShadows | None,
    rollout: WorldModelRollout | None,
    training_events: dict | None,
) -> None:
    """Drive the real robot and synchronize both shadows with the current observations."""
    observations = adapter.observations()
    history = observations["actor"].unsqueeze(1).repeat(1, actor.history_length, 1)
    frames, _ = adapter.motion_goal_window()
    velocity_commands = None
    if args.task_encoder == "velocity":
        velocity_commands = observations["actor"].new_empty((adapter.num_envs, 3))
        all_env_ids = torch.arange(adapter.num_envs, device=adapter.device)
        sample_velocity_commands(velocity_commands, velocity_cfg, args.velocity_command, all_env_ids)
    reset_playback_state(adapter, observations, rollout, training_events)

    env = adapter.env
    render_shadows = shadows is not None and env.sim.is_rendering
    if shadows is not None:
        shadows.update_reference_shadow()
        if rollout is not None:
            shadows.update_world_model_shadow(rollout)
    if render_shadows:
        env.cfg.sim.render_interval = (args.num_steps + 1) * env.cfg.decimation
        env.sim.render()

    for step in range(args.num_steps):
        goal = extract_actor_goal(observations["goal"], frames) if velocity_commands is None else velocity_commands
        action = actor(build_actor_obs_td({args.task_encoder: goal}, history))
        observations, _, terminated, truncated, _ = adapter.step(action)
        dones = terminated | truncated
        history = advance_proprio_history(history, observations["actor"], dones)
        if rollout is not None:
            rollout.advance_dynamics(action)
        if rollout is not None or (velocity_commands is not None and args.velocity_command is None):
            reset_ids = dones.nonzero(as_tuple=True)[0]
            if reset_ids.numel():
                reset_playback_state(adapter, observations, rollout, training_events, reset_ids)
                if velocity_commands is not None and args.velocity_command is None:
                    sample_velocity_commands(velocity_commands, velocity_cfg, None, reset_ids)
        if shadows is not None:
            shadows.update_reference_shadow()
            if rollout is not None:
                shadows.update_world_model_shadow(rollout)
        if render_shadows:
            env.sim.render()
        if step % 100 == 0:
            print(f"[SuperMimic] playback step={step}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--task-encoder", choices=("motion", "velocity"), default="motion")
    parser.add_argument("--velocity-command", nargs=3, type=float, metavar=("VX", "VY", "WZ"))
    parser.add_argument(
        "--show-reference-shadow",
        action=argparse.BooleanOptionalAction,
        help="Show the reference robot (Motion default).",
    )
    parser.add_argument(
        "--show-world-model-shadow",
        action=argparse.BooleanOptionalAction,
        help="Show the world model (Motion default).",
    )
    args, env_cfg, agent_cfg = parse_task_configs(parser, play=True)
    if args.num_steps < 0:
        parser.error("--num-steps must be non-negative")
    path = checkpoint_path(args, args.run_dir / agent_cfg.experiment_name, args.checkpoint or "latest")
    checkpoint = load_checkpoint(path, agent_cfg.representation, args.task_encoder)
    saved_config = checkpoint["config"]
    velocity_cfg = VelocityTaskCfg(**(agent_cfg.velocity.to_dict() | (saved_config.get("velocity") or {})))
    motion_playback = args.task_encoder == "motion"
    show_reference = motion_playback if args.show_reference_shadow is None else args.show_reference_shadow
    show_world_model = motion_playback if args.show_world_model_shadow is None else args.show_world_model_shadow
    if show_world_model and "dynamics" not in checkpoint:
        raise ValueError(
            "Checkpoint has no dynamics weights. Use --no-show-world-model-shadow for actor-only playback."
        )
    snapshot = {}
    if args.motion_file is None or show_world_model:
        snapshot = load_training_snapshot(path.parent / "params" / "env.yaml")
    load_training_motion_file(env_cfg, args.motion_file, snapshot)
    env_cfg.log_dir = str(path.parent)
    print(f"[SuperMimic] loading {path}")
    print(f"[SuperMimic] motion library: {env_cfg.commands.motion.motion_file}")

    with TemporaryDirectory(prefix="supermimic-shadows-") as directory, launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            env.reset()
            adapter = IsaacLabSuperMimicAdapter(env)
            actor = build_actor(adapter, checkpoint, args.task_encoder)
            if set(checkpoint["task_names"]) == {"motion", "velocity"}:
                exported = export_actor(actor, env, path.parent / "exported", path.stem, velocity_cfg)
                print("[SuperMimic] deployment files written to " + ", ".join(str(file) for file in exported))
            rollout = None
            if show_world_model:
                with torch.random.fork_rng(devices=[]):
                    world_model = build_world_model_from_state(
                        adapter,
                        checkpoint["representation"],
                        tuple(saved_config.get("dynamics_hidden_dims", agent_cfg.dynamics_hidden_dims)),
                    )
                world_model.dynamics.load_state_dict(checkpoint["dynamics"])
                rollout = WorldModelRollout(world_model.dynamics.eval(), world_model.representation)
            shadows = None
            if show_reference or show_world_model:
                shadows = PlaybackShadows(
                    adapter, Path(directory), show_reference=show_reference, show_world_model=show_world_model
                )
            run_playback(args, adapter, actor, velocity_cfg, shadows, rollout, snapshot.get("events"))
        finally:
            env.close()


if __name__ == "__main__":
    main()
