"""Train the AMP discriminator against imagined velocity-policy rollouts."""

import torch

from .runner import SuperAmpRunner


class ImaginedSuperAmpRunner(SuperAmpRunner):
    """Collect policy AMP windows from learned dynamics after each dynamics update."""

    def _init_sim_amp_history(self, full_state: torch.Tensor) -> None:
        """Imagined AMP history starts at each model rollout."""

    def _record_sim_amp(
        self,
        observed_state: torch.Tensor,
        next_state: torch.Tensor,
        reset_rows: torch.Tensor,
        next_state_valid: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Leave AMP replay collection to the imagined rollout before the policy update."""

    @torch.no_grad()
    def _collect_imagined_windows(self) -> None:
        env = self.world_model_env
        horizon = max(self.cfg.policy.horizon, self.amp_replay.window_length)
        starts = self.replay_buffer.sample_rollout(
            max(env.history_length, self.amp_replay.window_length - 1), horizon, self.cfg.policy.batch_size
        )
        observations = env.reset(starts)
        alive = torch.ones(env.batch_size, dtype=torch.bool, device=self.adapter.device)
        for _step in range(horizon):
            actions = self.actor(observations)
            actions = actions + torch.randn_like(actions) * self.cfg.action_noise
            observations, _, termination = env.step(actions)
            # Keep exactly the windows used by the AMP reward. The first sample
            # therefore contains the recorded history plus the first prediction.
            self.amp_replay.insert(env.last_amp_window[alive])
            alive = alive & (torch.rand_like(termination) >= termination)

    def _update_policy(self) -> dict[str, torch.Tensor]:
        self._collect_imagined_windows()
        return super()._update_policy()
