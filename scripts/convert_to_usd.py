"""Convert URDF and MuJoCo MJCF models to USD without launching Kit."""

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree


def _model_files(path: Path, suffix: str) -> list[Path]:
    """Collect files for the selected importer."""
    if path.is_file() and path.suffix.lower() == suffix:
        return [path]
    if path.is_dir():
        return sorted(file for file in path.rglob(f"*{suffix}", case_sensitive=False) if file.is_file())
    raise ValueError(f"expected a {suffix} file or a directory: {path}")


def main() -> int:
    """Convert the selected models and print their generated USD entry files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf_dir", type=Path, help="URDF file or directory to search recursively.")
    parser.add_argument("--xml_dir", type=Path, help="MuJoCo MJCF file or directory to search recursively.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory for the generated USD assets.")
    args = parser.parse_args()

    if args.urdf_dir is None and args.xml_dir is None:
        parser.error("provide --urdf_dir, --xml_dir, or both")
    output_dir = args.out_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"output path must be a directory: {output_dir}")

    urdf_files: list[Path] = []
    xml_files: list[Path] = []
    xml_input_is_file = False
    try:
        if args.urdf_dir is not None:
            urdf_files = _model_files(args.urdf_dir.expanduser().resolve(), ".urdf")
        if args.xml_dir is not None:
            xml_input = args.xml_dir.expanduser().resolve()
            xml_input_is_file = xml_input.is_file()
            xml_files = _model_files(xml_input, ".xml")
    except ValueError as exc:
        parser.error(str(exc))

    from isaaclab.utils.version import standalone_importers_available

    if not standalone_importers_available():
        parser.error("standalone importers are unavailable. Use a Kitless environment with 'uv run --extra importers'.")

    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg, UrdfConverter, UrdfConverterCfg

    converted = 0
    failed = 0
    for source in [*urdf_files, *xml_files]:
        try:
            if source.suffix.lower() == ".urdf":
                converter = UrdfConverter(
                    UrdfConverterCfg(
                        asset_path=str(source),
                        usd_dir=str(output_dir),
                        fix_base=False,
                        merge_fixed_joints=False,
                        force_usd_conversion=True,
                    )
                )
            else:
                if ElementTree.parse(source).getroot().tag != "mujoco":
                    if xml_input_is_file:
                        raise ValueError("expected a MuJoCo model with a <mujoco> root element")
                    continue
                converter = MjcfConverter(
                    MjcfConverterCfg(
                        asset_path=str(source),
                        usd_dir=str(output_dir),
                        import_physics_scene=False,
                        force_usd_conversion=True,
                    )
                )

            if not Path(converter.usd_path).is_file():
                raise RuntimeError("the importer did not generate a USD file")
        except Exception as exc:
            print(f"Failed to convert {source}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(converter.usd_path, flush=True)
        converted += 1

    if not converted and not failed:
        parser.error("no URDF or MuJoCo MJCF models found")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
