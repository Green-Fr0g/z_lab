"""Registered training configs for the G1 SuperMimic tasks."""

from isaaclab.utils import configclass

from z_lab.tasks.supermimic.rl_cfg import SuperMimicRunnerCfg


@configclass
class G1SuperMimicGenCoordRunnerCfg(SuperMimicRunnerCfg):
    """Train a G1 policy with generalized-coordinate dynamics."""

    experiment_name = "g1_supermimic_gencoord"
    representation = "gencoord"


@configclass
class G1SuperMimicMaxCoordRunnerCfg(SuperMimicRunnerCfg):
    """Train a G1 policy with maximal-coordinate dynamics."""

    experiment_name = "g1_supermimic_maxcoord"
    representation = "maxcoord"
