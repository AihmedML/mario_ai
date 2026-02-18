import torch
import numpy
from model import MarioNN, NUM_AUX


class MarioAgent:
    def __init__(self, num_actions, num_envs=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.num_envs = num_envs
        self.net = MarioNN(num_actions).to(self.device)

        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.1
        self.entropy_coeff_start = 0.05
        self.entropy_coeff_end = 0.005
        self.entropy_decay_updates = 1000
        self.entropy_coeff = self.entropy_coeff_start
        self.value_coeff = 0.5
        self.value_clip_epsilon = 0.2
        self.max_grad_norm = 0.5
        self.ppo_epochs = 3
        self.batch_size = 256
        self.rollout_length = 128

        # LR annealing
        self.max_lr = 1.5e-4
        self.total_updates = 3000
        self.update_count = 0

        self.target_kl = 0.012  # stop epoch early if policy changes too much
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.max_lr, eps=1e-5)
        self.step_count = 0

        # pre-allocated rollout buffer
        self.buf_states = numpy.zeros(
            (self.rollout_length, num_envs, 4, 84, 84), dtype=numpy.float32
        )
        self.buf_aux = numpy.zeros(
            (self.rollout_length, num_envs, NUM_AUX), dtype=numpy.float32
        )
        self.buf_actions = numpy.zeros(
            (self.rollout_length, num_envs), dtype=numpy.int64
        )
        self.buf_rewards = numpy.zeros(
            (self.rollout_length, num_envs), dtype=numpy.float32
        )
        self.buf_dones = numpy.zeros(
            (self.rollout_length, num_envs), dtype=numpy.float32
        )
        self.buf_log_probs = numpy.zeros(
            (self.rollout_length, num_envs), dtype=numpy.float32
        )
        self.buf_values = numpy.zeros(
            (self.rollout_length, num_envs), dtype=numpy.float32
        )
        self.buf_idx = 0
        self.current_aux = numpy.zeros((num_envs, NUM_AUX), dtype=numpy.float32)
        self.current_aux[:, 2] = 1.0
        self.last_x = numpy.zeros(num_envs, dtype=numpy.float32)

    @staticmethod
    def _explained_variance(y_true, y_pred):
        var_y = numpy.var(y_true)
        if var_y < 1e-8:
            return 0.0
        return float(1.0 - numpy.var(y_true - y_pred) / (var_y + 1e-8))

    @staticmethod
    def extract_aux(infos):
        """Extract normalized aux features from info dicts."""
        aux = numpy.zeros((len(infos), NUM_AUX), dtype=numpy.float32)
        for i, info in enumerate(infos):
            aux[i, 0] = numpy.clip(info.get("x_pos", 0) / 3200.0, 0.0, 1.2)
            aux[i, 1] = numpy.clip(info.get("y_pos", 0) / 240.0, 0.0, 1.2)
            aux[i, 2] = numpy.clip(info.get("time", 400) / 400.0, 0.0, 1.0)
        return aux

    def pick_action(self, states):
        """Takes batch of states (num_envs, 4, 84, 84), returns actions for all envs."""
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        aux_t = torch.as_tensor(self.current_aux, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, values = self.net(states_t, aux_t)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # store in buffer
        self.buf_states[self.buf_idx] = states
        self.buf_aux[self.buf_idx] = self.current_aux
        self.buf_actions[self.buf_idx] = actions.cpu().numpy()
        self.buf_log_probs[self.buf_idx] = log_probs.cpu().numpy()
        self.buf_values[self.buf_idx] = values.squeeze(-1).cpu().numpy()

        return actions.cpu().numpy()

    def remember(self, rewards, dones, infos):
        """Store rewards, dones, and update aux from infos."""
        shaped = rewards.astype(numpy.float32) / 15.0
        for i, info in enumerate(infos):
            x_pos = float(info.get("x_pos", 0))
            dx = max(0.0, x_pos - self.last_x[i])
            shaped[i] += dx * 0.01

            if info.get("flag_get", False):
                shaped[i] += 3.0
            elif dones[i]:
                shaped[i] -= 1.5

            self.last_x[i] = 0.0 if dones[i] else x_pos

        self.buf_rewards[self.buf_idx] = numpy.clip(shaped, -1.5, 3.0)
        self.buf_dones[self.buf_idx] = dones.astype(numpy.float32)
        self.buf_idx += 1
        self.step_count += self.num_envs
        self.current_aux = self.extract_aux(infos)
        done_mask = dones.astype(bool)
        if done_mask.any():
            self.current_aux[done_mask] = numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32)

    def ready_to_learn(self):
        return self.buf_idx >= self.rollout_length

    def learn(self, next_states):
        """PPO update. next_states needed for bootstrap value."""
        if not self.ready_to_learn():
            return None

        # LR annealing
        self.update_count += 1
        lr = self.max_lr * max(0.0, 1.0 - self.update_count / self.total_updates)
        lr = max(lr, 1e-6)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # Entropy annealing: start moderate, decay toward exploitation.
        progress = min(1.0, self.update_count / self.entropy_decay_updates)
        self.entropy_coeff = self.entropy_coeff_start + (
            self.entropy_coeff_end - self.entropy_coeff_start
        ) * progress

        # bootstrap value for GAE
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        aux_t = torch.as_tensor(self.current_aux, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, last_values = self.net(next_states_t, aux_t)
        last_values = last_values.squeeze(-1).cpu().numpy()

        # compute GAE advantages (vectorized across envs)
        advantages = numpy.zeros_like(self.buf_rewards)
        gae = numpy.zeros(self.num_envs)

        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_vals = last_values
            else:
                next_vals = self.buf_values[t + 1]

            delta = (
                self.buf_rewards[t]
                + self.gamma * next_vals * (1 - self.buf_dones[t])
                - self.buf_values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - self.buf_dones[t]) * gae
            advantages[t] = gae

        returns = advantages + self.buf_values
        explained_var = self._explained_variance(
            returns.reshape(-1), self.buf_values.reshape(-1)
        )

        # flatten (rollout_length * num_envs, ...)
        n = self.rollout_length * self.num_envs
        flat_states = torch.as_tensor(
            self.buf_states.reshape(n, 4, 84, 84), dtype=torch.float32, device=self.device
        )
        flat_aux = torch.as_tensor(
            self.buf_aux.reshape(n, NUM_AUX), dtype=torch.float32, device=self.device
        )
        flat_actions = torch.as_tensor(
            self.buf_actions.reshape(n), dtype=torch.int64, device=self.device
        )
        flat_old_log_probs = torch.as_tensor(
            self.buf_log_probs.reshape(n), dtype=torch.float32, device=self.device
        )
        flat_old_values = torch.as_tensor(
            self.buf_values.reshape(n), dtype=torch.float32, device=self.device
        )
        flat_advantages = torch.as_tensor(
            advantages.reshape(n), dtype=torch.float32, device=self.device
        )
        flat_returns = torch.as_tensor(
            returns.reshape(n), dtype=torch.float32, device=self.device
        )

        # normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )

        # PPO update
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        num_batches = 0
        early_stopped = False

        for epoch in range(self.ppo_epochs):
            if early_stopped:
                break
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                logits, values_pred = self.net(flat_states[idx], flat_aux[idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(flat_actions[idx])
                entropy = dist.entropy().mean()

                # KL early stopping: if policy changed too much, stop
                log_ratio = new_log_probs - flat_old_log_probs[idx]
                ratio = torch.exp(log_ratio)
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_frac = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()
                if approx_kl.item() > 1.5 * self.target_kl:
                    early_stopped = True
                    break

                clipped = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                )
                policy_loss = -torch.min(
                    ratio * flat_advantages[idx], clipped * flat_advantages[idx]
                ).mean()

                values_pred = values_pred.squeeze(-1)
                value_target = flat_returns[idx]
                value_pred_clipped = flat_old_values[idx] + torch.clamp(
                    values_pred - flat_old_values[idx],
                    -self.value_clip_epsilon,
                    self.value_clip_epsilon,
                )
                value_loss_unclipped = (values_pred - value_target).pow(2)
                value_loss_clipped = (value_pred_clipped - value_target).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    - self.entropy_coeff * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += approx_kl.item()
                total_clip_frac += clip_frac.item()
                num_batches += 1

        # reset buffer
        self.buf_idx = 0

        denom = max(num_batches, 1)
        return {
            "loss": total_loss / denom,
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
            "approx_kl": total_kl / denom,
            "clip_frac": total_clip_frac / denom,
            "explained_var": explained_var,
            "early_stopped": early_stopped,
            "lr": lr,
        }

    def save(self, path="mario_model.pth"):
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "update_count": self.update_count,
                "entropy_coeff": self.entropy_coeff,
                "entropy_coeff_start": self.entropy_coeff_start,
                "entropy_coeff_end": self.entropy_coeff_end,
                "entropy_decay_updates": self.entropy_decay_updates,
                "max_lr": self.max_lr,
                "last_x": self.last_x,
                "best_avg": getattr(self, "best_avg", 0),
            },
            path,
        )

    def load(self, path="mario_model.pth"):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint["net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint["step_count"]
        self.update_count = checkpoint["update_count"]
        self.max_lr = checkpoint.get("max_lr", self.max_lr)
        self.entropy_coeff_start = checkpoint.get(
            "entropy_coeff_start", self.entropy_coeff_start
        )
        self.entropy_coeff_end = checkpoint.get(
            "entropy_coeff_end", self.entropy_coeff_end
        )
        self.entropy_decay_updates = checkpoint.get(
            "entropy_decay_updates", self.entropy_decay_updates
        )
        self.entropy_coeff = checkpoint.get("entropy_coeff", self.entropy_coeff)
        if "last_x" in checkpoint:
            self.last_x = checkpoint["last_x"]
        self.best_avg = checkpoint.get("best_avg", 0)
