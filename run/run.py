"""
单进程
运行流程:
1. 创建环境, ac(agent), storage.
2. 采样 并 存入 storage
3. 更新
"""
import math
import numpy as np
import torch

from utils.env_utils import func_env_infos
from storage.storage import Storage
from algorithms.ppo import PPO

from parallel_env.parallel_env import make_parallel_envs
from collections import deque

from ac.ac import AC


def main(args):

    save_path = args.env_name + "_" + args.alg + ".pt"

    envs = make_parallel_envs(args.env_name, args.num_workers)
    obs_shape, action_space_shape, action_shape, action_type = func_env_infos(args.env_name)

    hidden_state_shape = obs_shape

    hidden_feature_shape = tuple(map(lambda x: x*3, obs_shape))

    actor_critic = AC(args.base_nn, obs_shape, hidden_state_shape, hidden_feature_shape,
                      action_space_shape, action_type)

    if args.alg == "ppo":
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
    else:
        raise NotImplementedError

    storage = Storage(obs_shape, action_shape, hidden_state_shape, args.num_workers,
                      args.num_steps)

    # : 所有workers, 0 第一步， 初始化状态必不为结束状态。
    # 此时，对应的mask应该为1-> done 为False
    storage.obs_vec[:, 0] = envs.reset()

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_workers
    reward_q = deque(maxlen=args.num_workers)

    for num_update_ in range(num_updates):
        for step_ in range(args.num_steps):  # 对应storage大小
            with torch.no_grad():  # 采样, h_n 为当前状态的h_n
                values, actions, h_ns, log_probs = actor_critic.act(*storage.retrive_act_data(step_))
            next_obs, rewards, dones, infos = envs.step(actions.numpy())  # 这里的数据已经排序完成

            masks = np.array([0. if done else 1. for done in dones])
            bad_masks = np.array([0. if 'bad_transition' in info.keys() else 1. for info in infos])  # 最大步数限制

            for info in infos:
                if "reward" in info.keys():
                    reward_q.append(info['reward'])

            storage.push(next_obs, actions.numpy(), h_ns.numpy(),
                         rewards, values.numpy(), log_probs.numpy(), masks, bad_masks)

        with torch.no_grad():
            # 获取storage中最后一个状态的value
            values = actor_critic.get_value(*storage.retrive_act_data(args.num_steps)).detach()

        storage.values_vec[:, -1] = values.numpy()

        alg.update(storage)

        storage.after_update()

        r_list = []
        if len(reward_q) == args.num_workers:
            while reward_q:
                r_list.append(reward_q.pop())

        if r_list:
            r_nd = np.array(r_list)
            print("Update times: {}, max reward: {}, min reward: {}, mean reward: {}".format(
                (num_update_+1), r_nd.max(), r_nd.min(), r_nd.mean())
            )

    envs.close()

    torch.save(actor_critic.state_dict(), save_path)
    print("done")


if __name__ == '__main__':
    import argparse
    import logging

    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="MountainCarContinuous-v0")
    # parser.add_argument("--env_name", type=str, default="CartPole-v0")

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--num_steps", type=int, default=10)

    parser.add_argument("--num_env_steps", type=int, default=20000000)

    parser.add_argument("--base_nn", type=str, default="mlp")

    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--mini_batch_size", type=int, default=10)

    parser.add_argument("--clip_eps", type=float, default=0.2)

    parser.add_argument("--critic_coef", type=float, default=0.5)

    parser.add_argument("--entropy_coef", type=float, default=0.01)

    parser.add_argument("--update_epochs", type=int, default=4)

    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--log_file", type=str, default="")

    parser.add_argument("--alg", type=str, default="ppo")

    args = parser.parse_args()

    # logging.basicConfig(filename=args.log_file, level=logging.INFO)

    main(args)
