"""
该模块用来定义存储模块:
由于所有更新相关数据都存放在这里, 所以应该有相应的计算方法.
定义类按照ppo算法定义, 该类也可以用于其他算法.

每一个实例保存多个actor采集到的样本.

"""

import numpy as np

from utils.transform import func_n2t
from torch.utils.data import BatchSampler, SubsetRandomSampler


class Storage(object):

    def __init__(self, obs_shape, action_shape, hidden_state_shape, num_workers, n_steps):
        """
        :param obs_shape: should be a tuple.
        :param action_shape:
        :param hidden_state_shape:
        :param num_workers:
        :param n_steps:
        """
        # 该属性用来保存观测值，预留出一个是为了在此基础上继续探索
        self.obs_vec = np.zeros((num_workers, n_steps+1, *obs_shape))
        self.hidden_states_vec = np.zeros((num_workers, n_steps+1, *hidden_state_shape))

        # 只有执行action才能获得reward
        self.actions_vec = np.zeros((num_workers, n_steps, *action_shape))

        self.rewards_vec = np.zeros((num_workers, n_steps, 1))

        # 这个log_prob可以通过.exp转换为prob.
        self.log_probs_vec = np.zeros((num_workers, n_steps, 1))

        # 这里是critic学习的value值
        self.values_vec = np.zeros((num_workers, n_steps+1, 1))

        # 使用reward计算得到的returns: r_t = r_{t+1} + ... + ...
        self.returns_vec = np.zeros((num_workers, n_steps+1, 1))

        # 用来记录该状态是否为结束状态, 0表示结束, 1表示正常
        self.masks_vec = np.ones((num_workers, n_steps+1, 1))

        # 用来记录该状态是否为结束状态, 0表示结束, 1表示正常, 到达指定步数
        self.bad_masks_vec = np.ones((num_workers, n_steps+1, 1))

        self.max_step = n_steps
        self.num_workers = num_workers

        self.step = 0

    def push(self, obs, action, hidden_state, reward, value, log_probs, masks, bad_masks):
        # 处理一下数据形式
        reward = reward.reshape(reward.shape[0], 1)
        masks = masks.reshape(masks.shape[0], 1)
        bad_masks = bad_masks.reshape(bad_masks.shape[0], 1)
        self.obs_vec[:, self.step+1] = obs
        self.actions_vec[:, self.step] = action
        self.hidden_states_vec[:, self.step+1] = hidden_state
        self.rewards_vec[:, self.step] = reward
        self.values_vec[:, self.step] = value
        self.log_probs_vec[:, self.step] = log_probs
        self.masks_vec[:, self.step+1] = masks
        self.bad_masks_vec[:, self.step+1] = bad_masks

        self.step = (self.step+1) % self.max_step

    def retrive_act_data(self, step):
        obs = self.obs_vec[:, step]
        hidden_state = self.hidden_states_vec[:, step]
        mask = self.masks_vec[:, step]

        return func_n2t((obs, hidden_state, mask))

    def after_update(self):
        self.obs_vec[:, 0] = self.obs_vec[:, -1]
        self.hidden_states_vec[:, 0] = self.hidden_states_vec[:, -1]
        self.masks_vec[:, 0] = self.masks_vec[:, -1]
        self.bad_masks_vec[:, 0] = self.bad_masks_vec[:, -1]

    def cal_returns(self,
                    gamma: float,
                    gae_lambda: float,
                    t_values: np.ndarray
                    ):
        """
        参考gae
        :param gamma:
        :param gae_lambda:
        :param t_values: 该值为规定步数内的最后一个状态的value, 应该是所有workers的.
        :return:
        """
        gae = np.zeros((self.num_workers, 1))
        self.values_vec[:, -1] = t_values

        for step_ in reversed(range(self.max_step)):
            # 如果下一步为终止状态，则其对应的V(s_{t+1})为0， 这里通过masks控制。
            delta = self.rewards_vec[:, step_] + \
                    gamma * self.values_vec[:, step_+1] * self.masks_vec[:, step_+1] - self.values_vec[:, step_]

            # 此处的masks作用是分离终止状态处的delta。
            gae = delta + gamma * gae_lambda * gae * self.masks_vec[:, step_+1]
            gae = gae * self.bad_masks_vec[:, step_+1]
            # 此处的returns即为A+V,其中gae为A
            self.returns_vec[:, step_] = gae + self.values_vec[:, step_]

    def sample_generator(self,
                         advantages_vec: np.ndarray,
                         mini_batch_size: int
                         ):

        samples_size = self.num_workers * self.max_step

        sampler = BatchSampler(
            SubsetRandomSampler(range(samples_size)),
            mini_batch_size,
            drop_last=True
        )  # tensor 作为索引可以直接应用在ndarray中.

        for indices in sampler:
            obs_batch = self.obs_vec[:, :-1].reshape(-1, *self.obs_vec.shape[2:])[indices]
            hidden_states_batch = self.hidden_states_vec[:, :-1].reshape(-1, *self.hidden_states_vec.shape[2:])[indices]
            actions_batch = self.actions_vec.reshape(-1, *self.actions_vec.shape[2:])[indices]

            values_batch = self.values_vec[:, :-1].reshape(-1, *self.values_vec.shape[2:])[indices]
            returns_batch = self.returns_vec[:, :-1].reshape(-1, *self.returns_vec.shape[2:])[indices]
            masks_batch = self.masks_vec[:, :-1].reshape(-1, *self.masks_vec.shape[2:])[indices]

            old_action_log_probs_batch = self.log_probs_vec.reshape(-1, *self.log_probs_vec.shape[2:])[indices]

            advantages_batch = advantages_vec.reshape(-1, *advantages_vec.shape[2:])[indices]

            yield obs_batch, hidden_states_batch, actions_batch, values_batch, returns_batch, \
                  masks_batch, old_action_log_probs_batch, advantages_batch


if __name__ == '__main__':
    import torch
    from utils.transform import func_n2t
    obs_shape = (1, )
    action_shape = (1, )
    hidden_shape = (1, )
    num_workers = 5
    n_steps = 10

    storage = Storage(obs_shape, action_shape, hidden_shape, num_workers, n_steps)
    storage.obs_vec[0, 0] = 1
    storage.push(0, 1, 1, 1, 1, 1, 1, 1)
    storage.push(0, 1, 1, 1, 1, 1, 1, 1)
    storage.push(1, 1, 1, 1, 1, 1, 1, 1)

    storage.cal_returns(0.99, 0.8, np.ones((5, 1)))
    # print(storage.returns_vec)
    # print(storage.obs_vec[0])
    advantages = storage.returns_vec - storage.values_vec[:, :-1]
    # print(torch.from_numpy(advantages).shape)

    s = storage.sample_generator(advantages, 5)
    for ss in s:
        obs, *a = ss
        # print(obs.shape)
        print(torch.from_numpy(obs).shape)

    # storage.after_update()
    # print(storage.obs_vec[0])