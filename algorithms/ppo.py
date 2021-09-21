"""
ppo algorithm.
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + (gamma * lambda) * delta_{t+1} + .. + .. + (gamma * lambda)^{T-t+1} * delta_{T-1}

ratio_t = pi_{theta}(a_t|s_t) / pi_{old}(a_t|s_t)
L^{clip}(theta) = E_t [min(ratio_t * A, clip(ratio_t, 1-eps, 1+eps)A_t]

L_t^{clip+VF+S) (theta) = E_t [L_t^{clip}(theta) - c1*L_t^{VF}(theta) + c2*S[pi(theta)(st)]

L_t^{VF} = (V_{theta} (st) - V_t^{target})**2
"""
import torch
import torch.optim as optim

from utils.transform import func_n2t, func_to


class PPO(object):

    def __init__(self,
                 actor_critic,
                 lr: float,
                 num_mini_batch: int,
                 clip_eps: float,
                 critic_coef: float,
                 entropy_coef: float,
                 update_epochs: int,
                 max_grad_norm: float,
                 eps=1e-5,
                 use_clipped_value_loss=True
                 ):
        self.actor_critic = actor_critic
        self.num_mini_batch = num_mini_batch

        self.clip_eps = clip_eps
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs

        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, storage, device, gamma=0.99, gae_lambda=0.95):
        """
        storage中的所有数据为ndarray格式. 每个数据的shape: (num_works, steps, *data_shape).
        :param storage:
        :param gamma:
        :param gae_lambda:
        :return:
        """
        storage.cal_returns(gamma, gae_lambda, storage.values_vec[:, -1])
        advantage_vec = storage.returns_vec[:, :-1] - storage.values_vec[:, :-1]
        # 标准化
        advantage_vec = (advantage_vec - advantage_vec.mean()) / (advantage_vec.std() + 1e-5)

        avg_action_loss = 0.
        avg_value_loss = 0.
        avg_dist_entropy = 0.

        for epoch in range(self.update_epochs):
            sample_generator = storage.sample_generator(advantage_vec, self.num_mini_batch)

            for sample in sample_generator:
                obs_batch, hidden_states_batch, actions_batch, values_batch, returns_batch, masks_batch, \
                old_action_log_probs_batch, advantages_batch = func_n2t(sample, device)

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_action(
                    obs_batch, hidden_states_batch, masks_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1. - self.clip_eps, 1. + self.clip_eps)*advantages_batch

                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = values_batch + \
                                         (values - values_batch).clamp(-self.clip_eps, self.clip_eps)
                    value_losses = (values - returns_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (returns_batch - values).pow(2).mean()
                loss = action_loss + self.critic_coef*value_loss - self.entropy_coef * dist_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                avg_action_loss += action_loss.item()
                avg_value_loss += value_loss.item()
                avg_dist_entropy += dist_entropy.item()

        num_updates = self.update_epochs * self.num_mini_batch
        avg_action_loss /= num_updates
        avg_value_loss /= num_updates
        avg_dist_entropy /= num_updates

        return avg_action_loss, avg_value_loss, avg_dist_entropy


