"""
定义一些用于获取环境相关信息的函数
"""
import gym
from environments.basic import func_generate_env


def func_env_infos(env_name):

    env = func_generate_env(env_name)

    action_space = env.action_space

    if action_space.__class__.__name__ == "Discrete":
        action_space_shape = tuple((action_space.n, ))
        action_shape = (1, )
        action_type = "discrete"
    elif action_space.__class__.__name__ == "Box":
        action_space_shape = action_space.shape
        action_shape = action_space_shape
        action_type = "continue"
    elif action_space.__class__.__name__ == "MultiBinary":
        action_space_shape = action_space.shape
        action_shape = action_space_shape
        action_type = "continue"
    else:
        raise NotImplementedError

    obs_shape = env.observation_space.shape

    env.close()

    return tuple(obs_shape), action_space_shape, action_shape, action_type
