"""Registered training configs for the DR02 SuperMimic tasks."""

from isaaclab.utils import configclass

from z_lab.tasks.supermimic.rl_cfg import SuperMimicRunnerCfg


@configclass
class DR02SuperMimicGenCoordRunnerCfg(SuperMimicRunnerCfg):
    """Train a DR02 policy with generalized-coordinate dynamics."""

    experiment_name = "dr02_supermimic_gencoord"
    representation = "gencoord"


@configclass
class DR02SuperMimicMaxCoordRunnerCfg(SuperMimicRunnerCfg):
    """Train a DR02 policy with maximal-coordinate dynamics."""

    experiment_name = "dr02_supermimic_maxcoord"
    representation = "maxcoord"
