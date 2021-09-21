"""
单进程
运行流程:
1. 创建环境, ac(agent), storage.
2. 采样 并 存入 storage
3. 更新
"""
import glob

import numpy as np
import torch

from environments.basic import func_generate_env
from utils.env_utils import func_env_infos


from ac.ac import AC


def main(args):
    # env_name_alg.pt 如 MountainCarContinuous-v0_ppo.pt

    pt = glob.glob("./pts/{}".format("*"+args.env_name+"*.pt"))[0]

    ob_shape, action_space_shape, action_shape, action_type = func_env_infos(args.env_name)
    env = func_generate_env(args.env_name)
    hidden_state_shape = ob_shape

    hidden_feature_shape = tuple(map(lambda x: x*6, ob_shape))

    actor_critic = AC(args.base_nn, ob_shape, hidden_state_shape, hidden_feature_shape,
                      action_space_shape, action_type)

    actor_critic.load_state_dict(torch.load(pt))

    ob = env.reset()
    h_n = torch.zeros(ob.shape)
    r = 0
    while True:
        with torch.no_grad():
            env.render()
            ob = torch.from_numpy(ob)
            value, action, h_n, log_prob = actor_critic.act(ob.float(), h_n, torch.ones(1))

            next_ob, reward, done, info = env.step(action.numpy())  # 这里的数据已经排序完成

            ob = next_ob
            h_n = h_n[0]
            r += reward

            if done:
                print(r)
                r = 0
                ob = env.reset()
    #
    # env.close()


if __name__ == '__main__':
    import argparse
    import logging

    parser = argparse.ArgumentParser()

    # parser.add_argument("--env_name", type=str, default="CartPole-v0")
    parser.add_argument("--env_name", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--base_nn", type=str, default="mlp")

    args = parser.parse_args()

    # logging.basicConfig(filename=args.log_file, level=logging.INFO)

    main(args)
