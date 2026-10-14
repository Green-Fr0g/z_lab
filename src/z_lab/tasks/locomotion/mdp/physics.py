

from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

@configclass
class FlatPhysicsCfg(PresetCfg):
    """Physics presets for humanoid locomotion on flat terrain."""

    isaacsim_physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.2
    )
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(
        isaacsim_physx=isaacsim_physx,
        ovphysx=ovphysx,
    )

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=65,
            nconmax=15,
            cone="pyramidal",
            update_data_interval=2,
            integrator="implicitfast",
            impratio=1.0,
        ),
        num_substeps=1,
        debug_mode=False,
    )

    default: NewtonCfg = newton_mjwarp

@configclass
class SimplePhysicsCfg(PresetCfg):
    """Physics presets for humanoid locomotion on simple terrain."""

    isaacsim_physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.2
    )
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
        collision_cfg=NewtonCollisionPipelineCfg(
            max_triangle_pairs=1_000_000,
        ),
        num_substeps=2,
        debug_mode=False,
    )

    default: NewtonCfg = newton_mjwarp

@configclass
class ComplexPhysicsCfg(PresetCfg):
    """Physics presets for humanoid locomotion on complex terrain."""

    isaacsim_physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_patch_count=10 * 2**15,
    )
    ovphysx: OvPhysxCfg = OvPhysxCfg(
        gpu_max_rigid_patch_count=10 * 2**15,
    )
    physx: PhysxAutoCfg = PhysxAutoCfg(
        isaacsim_physx=isaacsim_physx,
        ovphysx=ovphysx,
    )

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            update_data_interval=2,
            integrator="implicitfast",
            impratio=1.0,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            max_triangle_pairs=2_500_000,
        ),
        num_substeps=2,
        debug_mode=False,
    )

    default: NewtonCfg = newton_mjwarp
