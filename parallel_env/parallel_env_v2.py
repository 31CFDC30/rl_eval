"""
生成多个环境进行采样

1. 生成多个独立环境
2. 每个进程保持各自的环境
3. 进程接受不同的指令及数据, 并返回相应的数据.
4. 并行环境的使用方法, 应该与单个环境的使用方法相同或相似.
"""
import numpy as np
import multiprocessing as mp

from environments.basic import func_generate_env


class PEnv(object):
    # 应该接受一批pipe。
    # 数据流：每个pipe向另一端发送指令，并接受相应的返回值
    # reset：发送reset指令， 并获取每个env（eid）的初始化状态， 返回时， 应该已经是按照顺序打包好的数据
    # step： 发送step指令，及每个eid对应的action，所以， action需要按顺序发送， 返回时， 依旧是按顺序打包好的数据

    def __init__(self, pipe_list,):
        self.pipe_list = pipe_list

    def reset(self):
        for pipe in self.pipe_list:  # 这里应该是异步的？还是串行的？如果是串行的，那么似乎没必要使用多进程。
            self._reset(pipe)   # 创建进程似乎没必要，反而会增加没必要的进程调度花费
        obe_list = []

        # for pipe in self.pipe_list:
        #     obe_list.append(pipe.recv())
        #
        # ob_list = []
        # for i in range(len(obe_list)):
        #     ob_list.append(obe_list[i][1])

        for pipe in self.pipe_list:
            while True:
                com, data = pipe.recv()
                if com == 'state':
                    obe_list.append(data)
                    break

        return np.array(obe_list)

    def _reset(self, pipe):
        pipe.send(["reset", None])

    def step(self, actions: np.ndarray):
        # 这里的actions应该是已经按照eid排序好的。
        for eid, pipe in enumerate(self.pipe_list):  # 这里应该是异步的？还是串行的？如果是串行的，那么似乎没必要使用多进程。
            self._step(pipe, actions[eid])

        datae_list = []
        for pipe in self.pipe_list:
            while True:
                com, data = pipe.recv()
                if com == 'res':
                    datae_list.append(data)
                    break

        data_list = []
        next_ob_list = []
        reward_list = []
        done_list = []
        info_list = []
        for i in range(len(datae_list)):
            next_ob_list.append(datae_list[1][0])
            reward_list.append(datae_list[1][1])
            done_list.append(datae_list[1][2])
            info_list.append(datae_list[1][3])

        return np.array(next_ob_list), np.array(reward_list), np.array(done_list), np.array(info_list)

    def _step(self, pipe, action):
        pipe.send(["step", action])

    def close(self):
        for pipe in self.pipe_list:
            pipe.send(["close", None])

    def __len__(self):
        return len(self.pipe_list)


def worker(pipe, env, eid):

    def _step(env, action, r):
        # 如果done为True，那么下一个ob没有任何作用
        # 这里假设done针对的是next_ob, reward是在当前状态下执行相应动作得到的，info也是针对next_ob的，即这里的ob
        try:
            ob, reward, done, info = env.step(action)
        except Exception as e:
            ob, reward, done, info = env.step(action[0])
        if env._max_episode_steps == env._elapsed_steps:  # 超过最大步数
            info['bad_transition'] = True
        r += reward

        if done:
            # 下一个ob，即初始状态是否可以为done？
            # 如果为done，那么我们要做的只是将对应的hidden_state置0， hidden_state*mask=0
            # 其他部分则不需要额外操作。
            # 如果为结束状态，那么其对应的v(s)应该为0，在计算时，此处的值可以丢弃，所以可以直接reset。
            ob = env.reset()
            info["reward"] = r
            r = 0
        return ob, reward, done, info, r

    try:
        r = 0
        while True:
            cmd, data = pipe.recv()
            if cmd == "reset":
                ob = env.reset()
                pipe.send(('state', ob))
            elif cmd == "step":
                next_ob, reward, done, info, r = _step(env, data, r)
                pipe.send(('res', [next_ob, reward, done, info]))
            elif cmd == "render":
                env.render()
                pipe.send((eid, None))
            elif cmd == "close":
                pipe.close()
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("close")


def make_parallel_envs(env_name, num_workers):

    # 创建envs, eid为环境id
    envs = {eid: func_generate_env(env_name) for eid in range(num_workers)}

    # 为每一个环境创建一个pipe.
    # pipes = [mp.Pipe() for _ in range(num_workers)]
    pipes = [mp.Pipe(duplex=True) for _ in range(num_workers)]

    p_list = []

    pipe_list = []
    for ind in range(num_workers):
        p = mp.Process(target=worker, args=(pipes[ind][1], envs[ind], ind))
        p.start()
        p_list.append(p)
        pipe_list.append(pipes[ind][0])

    return PEnv(pipe_list)


if __name__ == '__main__':
    # 验证pipe是否两端都可以进行数据的接受与传送(可以)
    env = func_generate_env("CartPole-v0")

    envs = make_parallel_envs("CartPole-v0", 2)

    obs = envs.reset()

    while True:
        actions = []
        for i in range(len(envs)):
            actions.append(env.action_space.sample())

        actions = np.array(actions)

        obs, rewards, dones, infos = envs.step(actions)

        print(dones)









