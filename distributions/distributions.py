"""
针对ac结构定义常用的分布.
"""
import torch
import torch.nn as nn

from utils.initn import func_init
from torch.distributions import Normal

from functools import reduce


class FixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):

    def __init__(self,
                 input_shape: tuple,
                 output_shape: tuple
                 ):
        super(Categorical, self).__init__()
        init_ = lambda m: func_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0.),
                                    nn.init.calculate_gain("relu"))

        self.input_size = reduce(lambda x, y: x * y, input_shape)
        self.output_size = reduce(lambda x, y: x * y, output_shape)

        self.action_fc = init_(nn.Linear(self.input_size, self.output_size))

    def forward(self, x):
        x = x.view(-1, self.input_size)
        action = self.action_fc(x)
        return FixedCategorical(logits=action)


class FixedNormal(Normal):

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class Gaussian(nn.Module):

    def __init__(self,
                 input_shape: tuple,
                 output_shape: tuple
                 ):
        """
        :param input_shape:
        :param output_shape:
        """
        super(Gaussian, self).__init__()

        init_ = lambda m: func_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0.),
                                    nn.init.calculate_gain("tanh"))

        self.input_size = reduce(lambda x, y: x*y, input_shape)
        self.output_size = reduce(lambda x, y: x*y, output_shape)

        self.action_mean_fc = init_(nn.Linear(self.input_size, self.output_size))

        self.action_logstd = nn.Parameter(torch.randn(self.output_size), requires_grad=True)

    def forward(self, x):

        x = x.view(-1, self.input_size)

        action_mean = self.action_mean_fc(x)

        return FixedNormal(action_mean, self.action_logstd.exp())


if __name__ == '__main__':
    inputs = torch.randn((2, 3, 3))

    output_shape = (1, )

    # dist = Gaussian(tuple(inputs.shape[1:]), output_shape)
    dist = Categorical(tuple(inputs.shape[1:]), output_shape)

    print(dist(inputs).sample())