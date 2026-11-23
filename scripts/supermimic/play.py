"""Play a SuperMimic actor from a local checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import numpy as np
import torch
import warp as wp
import yaml
from cli_args import add_common_args, checkpoint_path, parse_task_configs
from isaaclab_visualizers.newton.newton_visualization_markers import NewtonVisualizationMarkers
from newton.viewer import ViewerBase, ViewerViser
from pxr import Usd, UsdGeom, UsdPhysics
from supermimic.envs.isaaclab_adapter import IsaacLabSuperMimicAdapter
from supermimic.envs.isaaclab_world_model import build_world_model_from_state
from supermimic.envs.world_model_env import WorldModelRollout
from supermimic.kinematics import ForwardKinematics, load_robot_model
from supermimic.models import MultiEncoderAE, TaskSpec
from supermimic.representations import PROPRIO_OBS_TERMS, extract_actor_goal
from supermimic.runners.exporter import export_actor
from supermimic.tasks import VelocityTaskCfg, advance_proprio_history, build_actor_obs_td
from supermimic.utils.math import matrix_to_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul

from isaaclab.app import launch_simulation
from isaaclab.envs.mdp import UniformVelocityCommandCfg, reset_scene_to_default
from isaaclab.managers import EventTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import PreviewSurfaceCfg, UsdFileCfg

from z_lab.tasks.locomotion.locomotion_env_cfg import TerminationsCfg as LocomotionTerminationsCfg
from z_lab.tasks.supermimic.tracking_env_cfg import SuperMimicGenCoordEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from z_lab.tasks.supermimic.rl_cfg import SuperMimicRunnerCfg


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


def build_actor(env: ManagerBasedRLEnv, checkpoint: dict, task_encoder: str) -> MultiEncoderAE:
    """Restore the actor using the robot's proprioception and checkpoint goal dimensions."""
    action_term = env.action_manager.get_term("joint_pos")
    if list(action_term._joint_names) != checkpoint["joint_names"]:
        raise ValueError("Checkpoint joints must match the playback task's action order")
    num_actions = action_term.action_dim
    observations = env.observation_manager
    term_dims = [(3,), (3,), (num_actions,), (num_actions,), (num_actions,)]
    if (
        tuple(observations.active_terms["actor"]) != PROPRIO_OBS_TERMS
        or observations.group_obs_term_dim["actor"] != term_dims
    ):
        raise ValueError("Unexpected SuperMimic actor observation terms or widths")
    for name in ("joint_pos", "joint_vel"):
        asset_cfg = getattr(env.cfg.observations.actor, name).params["asset_cfg"]
        _, joint_names = env.scene[asset_cfg.name].find_joints(
            asset_cfg.joint_names, preserve_order=asset_cfg.preserve_order
        )
        if list(joint_names) != checkpoint["joint_names"]:
            raise ValueError(f"Actor {name} joints must match the checkpoint joint order")

    saved_config = checkpoint["config"]
    goal_dims = {"velocity": 3}
    if "motion" in checkpoint["task_names"]:
        goal_dims["motion"] = saved_config["num_goal_frames"] * (2 * num_actions + 6)
    actor = MultiEncoderAE(
        obs_dim=env.obs_buf["actor"].shape[-1],
        action_dim=num_actions,
        task_specs=[TaskSpec(name, goal_dims[name]) for name in checkpoint["task_names"]],
        slot_mix={task_encoder: 1.0},
        latent_dim=saved_config["latent_dim"],
        hidden_dims=tuple(saved_config["actor_hidden_dims"]),
        history_length=saved_config["history_length"],
    ).to(env.device)
    actor.load_state_dict(checkpoint["actor"])
    return actor.eval()


def configure_motion_playback(env_cfg: SuperMimicGenCoordEnvCfg, motion_file: Path | None, snapshot: dict) -> None:
    """Use the full reference clip as the Motion episode."""
    training_motion = snapshot.get("commands", {}).get("motion", {}).get("motion_file")
    if motion_file is None and training_motion is not None:
        env_cfg.commands.motion.motion_file = training_motion
    env_cfg.commands.motion.debug_vis = False
    env_cfg.terminations.time_out = None
    env_cfg.terminations.anchor_pos = None
    env_cfg.terminations.anchor_ori = None


def configure_velocity_playback(
    env_cfg: SuperMimicGenCoordEnvCfg, velocity_cfg: VelocityTaskCfg, fixed_command: list[float] | None
) -> None:
    """Play from the default robot pose with timed velocity command sampling."""
    env_cfg.commands.motion = None
    env_cfg.observations.goal = None
    env_cfg.observations.full_state = None
    for name, term in vars(env_cfg.rewards).items():
        if term is not None and term.params.get("command_name") == "motion":
            setattr(env_cfg.rewards, name, None)
    env_cfg.events.reset_scene = EventTermCfg(func=reset_scene_to_default, mode="reset")

    ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=velocity_cfg.lin_vel_x_range,
        lin_vel_y=velocity_cfg.lin_vel_y_range,
        ang_vel_z=velocity_cfg.ang_vel_z_range,
    )
    if fixed_command is not None:
        forward, lateral, yaw_rate = fixed_command
        ranges.lin_vel_x = (forward, forward)
        ranges.lin_vel_y = (lateral, lateral)
        ranges.ang_vel_z = (yaw_rate, yaw_rate)
    env_cfg.commands.base_velocity = UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0), ranges=ranges, debug_vis=True
    )
    env_cfg.terminations = LocomotionTerminationsCfg()
    env_cfg.terminations.time_out = None


class _ShadowViewer:
    """Apply uniform opacity and Viser ghost lighting to marker draws."""

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
        viser_viewer = self._viewer if isinstance(self._viewer, ViewerViser) else None
        if viser_viewer is None and xforms is not None and not hidden:
            count = len(xforms)
            if count not in self._opacities:
                self._opacities[count] = wp.full(count, self._opacity, dtype=wp.float32, device=self._viewer.device)
            opacities = self._opacities[count]
        self._viewer.log_instances(name, mesh, xforms, scales, colors, materials, hidden=hidden, opacities=opacities)
        if viser_viewer is not None and xforms is not None and not hidden:
            handle = viser_viewer._scene_handles[viser_viewer._qualify(name)]
            if handle.opacity != self._opacity:
                handle.opacity = self._opacity
                handle.cast_shadow = False
                handle.receive_shadow = False


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
    name: str, meshes: dict[str, Path], color: tuple[float, float, float], *, opacity: float = 0.5
) -> VisualizationMarkers:
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
            _create_shadow_markers("world_model_shadow", meshes, (1.0, 0.45, 0.1), opacity=0.3)
            if show_world_model
            else None
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


@torch.no_grad()
def run_motion_playback(
    adapter: IsaacLabSuperMimicAdapter,
    actor: MultiEncoderAE,
    num_steps: int | None,
    shadows: PlaybackShadows | None,
    rollout: WorldModelRollout | None,
    training_events: dict | None,
) -> None:
    """Drive the real robot and synchronize both shadows with the current observations."""
    env = adapter.env
    motion_command = adapter.motion_command
    if num_steps is None:
        start_frames = motion_command.time_steps
        num_steps = int(motion_command.motion.clip_end(start_frames)[0] - start_frames[0])
    observations = adapter.observations()
    history = observations["actor"].unsqueeze(1).repeat(1, actor.history_length, 1)
    frames, _ = adapter.motion_goal_window()
    if rollout is not None:
        rollout.reset_dynamics(
            observations["full_state"], adapter.sim_context(training_events), adapter.previous_actions()
        )

    render_shadows = shadows is not None and env.sim.is_rendering
    if shadows is not None:
        shadows.update_reference_shadow()
        if rollout is not None:
            shadows.update_world_model_shadow(rollout)
    if render_shadows:
        env.cfg.sim.render_interval = (num_steps + 1) * env.cfg.decimation
        env.sim.render()

    for step in range(num_steps):
        goal = extract_actor_goal(observations["goal"], frames)
        action = actor(build_actor_obs_td({"motion": goal}, history))
        if rollout is not None:
            rollout.reset_dynamics(
                observations["full_state"], adapter.sim_context(training_events), adapter.previous_actions()
            )
            rollout.advance_dynamics(action)
        observations, _, terminated, truncated, _ = adapter.step(action)
        dones = terminated | truncated
        history = advance_proprio_history(history, observations["actor"], dones)
        if rollout is not None:
            reset_ids = dones.nonzero(as_tuple=True)[0]
            if reset_ids.numel():
                rollout.reset_dynamics(
                    observations["full_state"],
                    adapter.sim_context(training_events),
                    adapter.previous_actions(),
                    reset_ids,
                )
        if shadows is not None:
            shadows.update_reference_shadow()
            if rollout is not None:
                shadows.update_world_model_shadow(rollout)
        if render_shadows:
            env.sim.render()
        if step % 100 == 0:
            print(f"[SuperMimic] playback step={step}")


@torch.no_grad()
def run_velocity_playback(env: ManagerBasedRLEnv, actor: MultiEncoderAE, num_steps: int | None) -> None:
    """Drive the actor with the command manager's current body-frame velocity target."""
    command = env.command_manager.get_term("base_velocity")
    command.compute(dt=0.0)
    history = env.obs_buf["actor"].unsqueeze(1).repeat(1, actor.history_length, 1)
    step = 0
    while (num_steps is None or step < num_steps) and env.sim.is_running():
        action = actor(build_actor_obs_td({"velocity": command.command}, history))
        observations, _, terminated, truncated, _ = env.step(action)
        history = advance_proprio_history(history, observations["actor"], terminated | truncated)
        if step % 100 == 0:
            print(f"[SuperMimic] playback step={step}")
        step += 1


def play_motion(
    env: ManagerBasedRLEnv,
    args: argparse.Namespace,
    path: Path,
    checkpoint: dict,
    velocity_cfg: VelocityTaskCfg,
    snapshot: dict,
    dynamics_hidden_dims: tuple[int, ...],
) -> None:
    """Validate the Motion layout, export the policy, and prepare the selected shadows."""
    adapter = IsaacLabSuperMimicAdapter(env)
    adapter.validate_proprio_layout()
    adapter.validate_full_state_layout(checkpoint["representation"], env.obs_buf["full_state"].shape[-1])
    if list(adapter.motion_command.cfg.body_names) != checkpoint["body_names"]:
        raise ValueError("Checkpoint links must match the playback task's tracking order")
    saved_config = checkpoint["config"]
    if adapter.motion_goal_window() != (saved_config["num_goal_frames"], saved_config["goal_frame_stride"]):
        raise ValueError("Checkpoint Motion window must match the playback task")
    actor = build_actor(env, checkpoint, "motion")
    if set(checkpoint["task_names"]) == {"motion", "velocity"}:
        exported = export_actor(actor, env, path.parent / "exported", path.stem, velocity_cfg)
        print("[SuperMimic] deployment files written to " + ", ".join(str(file) for file in exported))

    rollout = None
    if args.show_world_model_shadow:
        with torch.random.fork_rng(devices=[]):
            world_model = build_world_model_from_state(adapter, checkpoint["representation"], dynamics_hidden_dims)
        world_model.dynamics.load_state_dict(checkpoint["dynamics"])
        rollout = WorldModelRollout(world_model.dynamics.eval(), world_model.representation)

    if args.show_reference_shadow or rollout is not None:
        with TemporaryDirectory(prefix="supermimic-shadows-") as directory:
            shadows = PlaybackShadows(
                adapter,
                Path(directory),
                show_reference=args.show_reference_shadow,
                show_world_model=rollout is not None,
            )
            run_motion_playback(adapter, actor, args.num_steps, shadows, rollout, snapshot.get("events"))
    else:
        run_motion_playback(adapter, actor, args.num_steps, None, None, None)


def parse_playback_args() -> tuple[argparse.Namespace, SuperMimicGenCoordEnvCfg, SuperMimicRunnerCfg]:
    """Parse playback flags and resolve the registered task and agent configs."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    parser.add_argument(
        "--num-steps", type=int, help="Playback step limit. Defaults to one Motion clip or unlimited Velocity playback."
    )
    parser.add_argument("--task-encoder", choices=("motion", "velocity"), default="motion")
    parser.add_argument("--velocity-command", nargs=3, type=float, metavar=("VX", "VY", "WZ"))
    parser.add_argument(
        "--show-reference-shadow",
        action=argparse.BooleanOptionalAction,
        help="Show the reference robot in Motion playback (enabled by default).",
    )
    parser.add_argument(
        "--show-world-model-shadow",
        action=argparse.BooleanOptionalAction,
        help="Show the world model in Motion playback (enabled by default).",
    )
    args, env_cfg, agent_cfg = parse_task_configs(parser, play=True)
    if args.num_steps is not None and args.num_steps < 0:
        parser.error("--num-steps must be non-negative")
    if args.task_encoder == "velocity" and (args.show_reference_shadow or args.show_world_model_shadow):
        parser.error("Shadow display is available in Motion playback")
    if args.task_encoder == "motion":
        args.show_reference_shadow = args.show_reference_shadow is not False
        args.show_world_model_shadow = args.show_world_model_shadow is not False
    return args, env_cfg, agent_cfg


def main() -> None:
    args, env_cfg, agent_cfg = parse_playback_args()
    path = checkpoint_path(args, args.run_dir / agent_cfg.experiment_name, args.checkpoint or "latest")
    checkpoint = load_checkpoint(path, agent_cfg.representation, args.task_encoder)
    velocity_cfg = VelocityTaskCfg(**(agent_cfg.velocity.to_dict() | (checkpoint["config"].get("velocity") or {})))
    if args.show_world_model_shadow and "dynamics" not in checkpoint:
        raise ValueError(
            "Checkpoint has no dynamics weights. Use --no-show-world-model-shadow for actor-only playback."
        )
    snapshot = {}
    env_cfg.log_dir = str(path.parent)
    print(f"[SuperMimic] loading {path}")
    if args.task_encoder == "motion":
        if args.motion_file is None or args.show_world_model_shadow:
            snapshot = load_training_snapshot(path.parent / "params" / "env.yaml")
        configure_motion_playback(env_cfg, args.motion_file, snapshot)
        print(f"[SuperMimic] motion library: {env_cfg.commands.motion.motion_file}")
    else:
        configure_velocity_playback(env_cfg, velocity_cfg, args.velocity_command)

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            env.reset()
            if args.task_encoder == "motion":
                hidden_dims = tuple(checkpoint["config"].get("dynamics_hidden_dims", agent_cfg.dynamics_hidden_dims))
                play_motion(env.unwrapped, args, path, checkpoint, velocity_cfg, snapshot, hidden_dims)
            else:
                actor = build_actor(env.unwrapped, checkpoint, "velocity")
                run_velocity_playback(env.unwrapped, actor, args.num_steps)
        finally:
            env.close()


if __name__ == "__main__":
    main()
