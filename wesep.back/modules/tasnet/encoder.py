import torch.nn as nn

from wesep.modules.tasnet.convs import Conv1D


class DeepEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DeepEncoder, self).__init__()
        self.sequential = nn.Sequential(
            Conv1D(in_channels, out_channels, kernel_size, stride=stride),
            Conv1D(out_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(),
            Conv1D(out_channels, out_channels, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.PReLU(),
            Conv1D(out_channels, out_channels, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.PReLU(),
            Conv1D(out_channels, out_channels, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.PReLU()
        )

    def forward(self, x):
        """
        :param  x: [B, T]
        :return: out: [B, N, T]
        """

        x = self.sequential(x)
        return x
