"""Registered training configs for the G1 23-DOF SuperAmp tasks."""

from isaaclab.utils import configclass

from z_lab.tasks.superamp.rl_cfg import SuperAmpRunnerCfg


@configclass
class G1_23DOFSuperAmpGenCoordRunnerCfg(SuperAmpRunnerCfg):
    """Train a G1 23-DOF policy with generalized-coordinate dynamics."""

    experiment_name = "g1_23dof_superamp_gencoord"
    representation = "gencoord"


@configclass
class G1_23DOFSuperAmpMaxCoordRunnerCfg(SuperAmpRunnerCfg):
    """Train a G1 23-DOF policy with maximal-coordinate dynamics."""

    experiment_name = "g1_23dof_superamp_maxcoord"
    representation = "maxcoord"
