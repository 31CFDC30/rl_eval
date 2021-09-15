"""
单进程
运行流程:
1. 创建环境, ac(agent), storage.
2. 采样 并 存入 storage
3. 更新
"""
import numpy as np
import torch

from environments.basic import func_generate_env
from utils.env_utils import func_env_infos
from storage.storage import Storage
from algorithms.ppo import PPO

from parallel_env.parallel_env import make_parallel_envs

from ac.ac import AC


def main(args):

    env = func_generate_env(args.env_name)
    envs = make_parallel_envs(args.env_name, args.num_workers)
    obs_shape, action_shape, action_type = func_env_infos(env)

    hidden_state_shape = obs_shape

    hidden_feature_shape = tuple(map(lambda x: x*2, obs_shape))

    actor_critic = AC(args.base_nn, obs_shape, hidden_state_shape, hidden_feature_shape,
                      action_shape, action_type)

    # actor_critic.load_state_dict(torch.load("test.pt"))

    alg = PPO(
        actor_critic,
        args.lr,
        args.mini_batch_size,
        args.clip_eps,
        args.critic_coef,
        args.entropy_coef,
        args.update_epochs,
        args.max_grad_norm
    )

    storage = Storage(obs_shape, action_shape, hidden_state_shape, args.num_workers,
                      args.num_steps)

    storage.obs_vec[:, 0] = envs.reset()  # : 所有workers, 0 第一步.

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_workers

    for num_update_ in range(num_updates):

        for step_ in range(args.num_steps):
            # env.render()
            with torch.no_grad():  # 采样, h_n 为当前状态的h_n
                values, actions, h_ns, log_probs = actor_critic.act(*storage.retrive_act_data(step_))

            next_obs, rewards, dones, infos = envs.step(actions.numpy())  # 这里的数据已经排序完成

            masks = np.array([0. if done else 1. for done in dones])

            storage.push(next_obs, actions.numpy(), h_ns.numpy(),
                         rewards, values.numpy(), log_probs.numpy(), masks)

        with torch.no_grad():
            values = actor_critic.get_value(*storage.retrive_act_data(args.num_steps))

        storage.values_vec[:, -1] = values.numpy()
        alg.update(storage)

        storage.after_update()

        print(num_update_, storage.rewards_vec.mean())

    env.close()

    torch.save(actor_critic.state_dict(), "test.pt")


if __name__ == '__main__':
    import argparse
    import logging

    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="MountainCarContinuous-v0")

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--num_steps", type=int, default=10)

    parser.add_argument("--num_env_steps", type=int, default=1000000)

    parser.add_argument("--base_nn", type=str, default="mlp")

    parser.add_argument("--lr", type=float, default=7e-4)

    parser.add_argument("--mini_batch_size", type=int, default=5)

    parser.add_argument("--clip_eps", type=float, default=0.2)

    parser.add_argument("--critic_coef", type=float, default=0.5)

    parser.add_argument("--entropy_coef", type=float, default=0.01)

    parser.add_argument("--update_epochs", type=int, default=5)

    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--log_file", type=str, default="/home/catsled/RL/log/1.log")

    args = parser.parse_args()

    # logging.basicConfig(filename=args.log_file, level=logging.INFO)

    main(args)
