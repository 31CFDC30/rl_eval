"""
多层感知机MLP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from utils.initn import func_init


class MLP(nn.Module):

    def __init__(self,
                 input_shape: tuple,
                 hidden_state_shape: tuple,
                 output_shape: tuple,
                 ):
        """
        input_shape 对应 obs_shape
        output_shape 对应 hidden_x_shape
        hidden_state_shape 对应 gru, h_n
        :param input_shape:
        :param output_shape:
        :param hidden_state_shape
        """
        super(MLP, self).__init__()

        init_ = lambda m: func_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0.),
                                    nn.init.calculate_gain("tanh"))

        self.input_size = reduce(lambda x, y: x*y, input_shape)
        self.output_size = reduce(lambda x, y: x*y, output_shape)
        self.hidden_state_size = reduce(lambda x, y: x*y, hidden_state_shape)

        self.gru = nn.GRU(self.input_size, self.hidden_state_size)

        self.action_fc = nn.Sequential(
            init_(nn.Linear(self.hidden_state_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh()
        )

        self.value_fc = nn.Sequential(
            init_(nn.Linear(self.hidden_state_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh()
        )

        self.value_head = init_(nn.Linear(self.output_size, 1))

    def forward(self, x, hidden_state, masks):
        # 参照gru输入格式
        if x.size(0) == hidden_state.size(0):
            x = x.view(-1, self.input_size).unsqueeze(0)
            hidden_state = hidden_state.view(-1, self.hidden_state_size).unsqueeze(0)
        else:
            raise NotImplemented

        hidden_x, h_n = self.gru(x, hidden_state*masks.unsqueeze(0))
        hidden_x = hidden_x.squeeze(0)
        h_n = h_n.squeeze(0)

        a_x = self.action_fc(hidden_x)
        v_x = self.value_fc(hidden_x)
        value = self.value_head(v_x)

        return a_x, h_n, value


class MLP2(nn.Module):

    def __init__(self,
                 input_shape: tuple,
                 hidden_state_shape: tuple,
                 output_shape: tuple,
                 ):
        """
        input_shape 对应 obs_shape
        output_shape 对应 hidden_x_shape
        hidden_state_shape 对应 gru, h_n
        :param input_shape:
        :param output_shape:
        :param hidden_state_shape
        """
        super(MLP2, self).__init__()

        init_ = lambda m: func_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0.),
                                    nn.init.calculate_gain("tanh"))

        self.input_size = reduce(lambda x, y: x*y, input_shape)
        self.output_size = reduce(lambda x, y: x*y, output_shape)
        self.hidden_state_size = reduce(lambda x, y: x*y, hidden_state_shape)

        self.action_fc = nn.Sequential(
            init_(nn.Linear(self.input_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh()
        )

        self.value_fc = nn.Sequential(
            init_(nn.Linear(self.input_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh()
        )

        self.value_head = init_(nn.Linear(self.output_size, 1))

    def forward(self, x, hidden_state, masks):

        a_x = self.action_fc(x)
        v_x = self.value_fc(x)
        value = self.value_head(v_x)

        return a_x, hidden_state, value


if __name__ == '__main__':
    mlp = MLP((3, 3), (2, 2), (3, 3))
    print(mlp)

    x, h_n, value = mlp(torch.randn((4, 3, 3)), torch.randn((4, 2, 2)), torch.ones((4, 1)))
    print(x.shape, h_n.shape, value)