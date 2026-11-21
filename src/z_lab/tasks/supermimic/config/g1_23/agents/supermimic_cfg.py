"""Registered training configs for the G1 23-DOF SuperMimic tasks."""

from isaaclab.utils import configclass

from z_lab.tasks.supermimic.rl_cfg import SuperMimicRunnerCfg


@configclass
class G1_23DOFSuperMimicGenCoordRunnerCfg(SuperMimicRunnerCfg):
    """Train a G1 23-DOF policy with generalized-coordinate dynamics."""

    experiment_name = "g1_23dof_supermimic_gencoord"
    representation = "gencoord"


@configclass
class G1_23DOFSuperMimicMaxCoordRunnerCfg(SuperMimicRunnerCfg):
    """Train a G1 23-DOF policy with maximal-coordinate dynamics."""

    experiment_name = "g1_23dof_supermimic_maxcoord"
    representation = "maxcoord"
