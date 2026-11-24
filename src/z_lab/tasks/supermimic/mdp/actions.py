"""Joint position control through biased encoders."""

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class BiasedJointPositionAction(JointPositionAction):
    """Apply position targets in the joint coordinates reported by the encoders."""

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        bias = getattr(self._env, "supermimic_encoder_bias", None)
        if bias is not None:
            self.processed_actions.sub_(bias[:, self._joint_ids])


@configclass
class BiasedJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action with a startup encoder offset."""

    class_type: type = BiasedJointPositionAction
