"""Physics presets for Mimic environments."""

from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg


@configclass
class SimplePhysicsCfg(PresetCfg):
    """Physics presets for Mimic on a simple plane scene."""

    isaacsim_physx: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(
        isaacsim_physx=isaacsim_physx,
        ovphysx=ovphysx,
    )

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=300,
            nconmax=80,
            cone="pyramidal",
            update_data_interval=2,
            integrator="implicitfast",
            impratio=1.0,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=1_000_000),
        num_substeps=2,
        debug_mode=False,
    )

    default: NewtonCfg = newton_mjwarp
