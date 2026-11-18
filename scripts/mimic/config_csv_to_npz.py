"""Read a motion config yaml and convert its CSV paths via ``csv_to_npz.py``.

This script is the config-driven batch front end. It reuses the conversion
functions from ``csv_to_npz.py`` and keeps one Isaac Sim session for all CSVs:

1. read the robot and motion metadata from a yaml config
2. resolve every CSV path relative to that config file
3. convert all CSVs in one initialized simulation session

Example:
    python scripts/mimic/config_csv_to_npz.py \
        --config motion_data/config/g1.yaml \
        --output_dir /tmp/motions

    python scripts/mimic/config_csv_to_npz.py \
        --config motion_data/config/g1.yaml \
        --output_dir /tmp/motions \
        --overwrite --headless
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from csv_to_npz import (
    InteractiveScene,
    PhysicsCfg,
    SimulationContext,
    _build_scene_cfg,
    _get_robot_asset,
    add_launcher_args,
    convert_motion,
    launch_simulation,
    sim_utils,
)


@dataclass(frozen=True)
class MotionConfig:
    """Robot and motion metadata read from a converter config."""

    robot_name: str
    root_link_name: str
    root_quat_order: str
    input_fps: int
    csv_joint_names: tuple[str, ...]
    csv_paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a motion config yaml and convert its CSV paths via scripts/mimic/csv_to_npz.py."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Motion config yaml containing a motion_data mapping.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the generated NPZ files will be written.",
    )
    parser.add_argument("--output_fps", type=int, default=50, help="Output NPZ frame rate.")
    parser.add_argument(
        "--physics",
        choices=("physx", "isaacsim_physx", "newton_mjwarp", "newton_vbd", "ovphysx"),
        default="newton_mjwarp",
        help="Physics backend passed to the Isaac Lab 3.0 converter.",
    )
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional 1-based inclusive frame range passed to the single-file converter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate NPZ files even when the output already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the batch plan without starting simulation or writing files.",
    )
    parser.add_argument(
        "--visual",
        choices=("newton_gl", "newton_rtx", "kit", "rerun", "viser", "none"),
        default="none",
        help="Visualizer backend. Defaults to headless conversion.",
    )
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.headless:
        args.visual = "none"
    return args


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml structure in: {path}")
    return data


def _resolve_path(base_dir: Path, maybe_path: str) -> Path:
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _require_str(motion_cfg: dict[str, Any], key: str, config_path: Path) -> str:
    value = motion_cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"motion_data.{key} must be a non-empty string in: {config_path}")
    return value


def _require_str_list(motion_cfg: dict[str, Any], key: str, config_path: Path) -> list[str]:
    value = motion_cfg.get(key)
    if not isinstance(value, list) or not value:
        raise TypeError(f"motion_data.{key} must be a non-empty list of strings in: {config_path}")
    entries: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"motion_data.{key} must contain only non-empty strings in: {config_path}")
        entries.append(item)
    return entries


def load_motion_config(config_path: Path) -> MotionConfig:
    """Validate a converter config and return its motion metadata."""
    config = _load_yaml(config_path)
    motion_cfg = config.get("motion_data")
    if not isinstance(motion_cfg, dict):
        raise TypeError(f"'motion_data' must be a mapping in: {config_path}")

    root_quat_order = str(motion_cfg.get("root_quat_order", "xyzw"))
    if root_quat_order not in ("wxyz", "xyzw"):
        raise ValueError(f"motion_data.root_quat_order must be 'wxyz' or 'xyzw' in: {config_path}")

    input_fps = int(motion_cfg.get("csv_fps", 30))
    if input_fps <= 0:
        raise ValueError(f"motion_data.csv_fps must be positive in: {config_path}")

    return MotionConfig(
        robot_name=_require_str(motion_cfg, "robot_name", config_path),
        root_link_name=_require_str(motion_cfg, "root_link_name", config_path),
        root_quat_order=root_quat_order,
        input_fps=input_fps,
        csv_joint_names=tuple(_require_str_list(motion_cfg, "csv_joint_names", config_path)),
        csv_paths=tuple(
            _resolve_path(config_path.parent, entry)
            for entry in _require_str_list(motion_cfg, "csv_paths", config_path)
        ),
    )


def _unique_output_path(output_dir: Path, csv_path: Path, used_names: set[str]) -> Path:
    """Pick a stable, collision-free NPZ name for one CSV file."""
    stem = csv_path.stem
    candidate = f"{stem}.npz"
    if candidate not in used_names:
        used_names.add(candidate)
        return output_dir / candidate

    parent_stem = csv_path.parent.name or "motion"
    candidate = f"{parent_stem}__{stem}.npz"
    if candidate not in used_names:
        used_names.add(candidate)
        return output_dir / candidate

    index = 2
    while True:
        candidate = f"{parent_stem}__{stem}_{index}.npz"
        if candidate not in used_names:
            used_names.add(candidate)
            return output_dir / candidate
        index += 1


def _print_line(message: str) -> None:
    """Print one status line immediately so parent and child output stay ordered."""
    print(message, flush=True)


def main() -> None:
    args = parse_args()

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    motion_cfg = load_motion_config(config_path)
    if args.output_fps <= 0:
        raise ValueError("--output_fps must be positive")
    if args.frame_range is not None and (args.frame_range[0] < 1 or args.frame_range[1] < args.frame_range[0]):
        raise ValueError("--frame_range must be a 1-based inclusive range with START <= END")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "DRY RUN" if args.dry_run else "CONVERT"
    _print_line(
        f"[{mode}] {config_path.name} | robot={motion_cfg.robot_name} | "
        f"jobs={len(motion_cfg.csv_paths)} | {motion_cfg.input_fps} -> {args.output_fps} FPS"
    )
    _print_line(f"Output: {output_dir}")

    converted = 0
    skipped = 0
    failed = 0
    missing = 0
    used_names: set[str] = set()
    jobs: list[tuple[int, Path, Path, str]] = []

    for index, csv_path in enumerate(motion_cfg.csv_paths, start=1):
        output_path = _unique_output_path(output_dir, csv_path, used_names)
        action_name = csv_path.stem

        if output_path.exists() and not args.overwrite:
            if not args.dry_run:
                _print_line(f"[{index:02d}/{len(motion_cfg.csv_paths):02d}] SKIP {action_name}")
            skipped += 1
            continue

        if not csv_path.is_file():
            missing += 1
            if not args.dry_run:
                _print_line(f"[{index:02d}/{len(motion_cfg.csv_paths):02d}] FAIL {action_name} | input missing")
            continue
        jobs.append((index, csv_path, output_path, action_name))

    if args.dry_run:
        planned = len(jobs)
        if args.verbose:
            for index, _, output_path, action_name in jobs:
                _print_line(f"[{index:02d}/{len(motion_cfg.csv_paths):02d}] PLAN {action_name} -> {output_path.name}")
        _print_line(f"[DONE] planned={planned} | skipped={skipped} | missing={missing}")
        return

    if not jobs:
        _print_line(f"[DONE] generated=0 | skipped={skipped} | missing={missing} | failed=0")
        if missing > 0:
            raise SystemExit(1)
        return

    robot_cfg = _get_robot_asset(motion_cfg.robot_name)
    args.visualizer = [] if args.visual == "none" else [args.visual]
    sim_cfg = sim_utils.SimulationCfg(device=args.device, dt=1.0 / args.output_fps)
    sim_cfg.physics = PhysicsCfg()
    with launch_simulation(sim_cfg, args) as physics_cfg:
        sim_cfg.physics = physics_cfg
        sim = SimulationContext(sim_cfg)
        scene_cfg = _build_scene_cfg(robot_cfg)
        scene = InteractiveScene(scene_cfg(num_envs=1, env_spacing=2.0))
        sim.reset()

        for index, csv_path, output_path, action_name in jobs:
            _print_line(f"[{index:02d}/{len(motion_cfg.csv_paths):02d}] RUN  {action_name}")
            try:
                convert_motion(
                    sim=sim,
                    scene=scene,
                    input_file=csv_path,
                    output_file=output_path,
                    input_fps=motion_cfg.input_fps,
                    output_fps=args.output_fps,
                    csv_joint_names=list(motion_cfg.csv_joint_names),
                    root_link_name=motion_cfg.root_link_name,
                    robot_name=motion_cfg.robot_name,
                    root_quat_order=motion_cfg.root_quat_order,
                    frame_range=tuple(args.frame_range) if args.frame_range is not None else None,
                    verbose=args.verbose,
                )
            except Exception as exc:
                _print_line(f"[{index:02d}/{len(motion_cfg.csv_paths):02d}] FAIL {action_name} | {exc}")
                failed += 1
                continue
            _print_line(f"[{index:02d}/{len(motion_cfg.csv_paths):02d}] OK   {action_name}")
            converted += 1

    _print_line(f"[DONE] generated={converted} | skipped={skipped} | missing={missing} | failed={failed}")
    if failed > 0 or missing > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
