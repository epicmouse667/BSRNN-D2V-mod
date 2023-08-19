import torch
import torch.nn as nn


class DeepDecoder(nn.Module):
    """
        Decoder
        This module can be seen as the gradient of Conv1d with respect to its input.
        It is also known as a fractionally-strided convolution
        or a deconvolution (although it is not an actual deconvolution operation).
    """

    def __init__(self, N, kernel_size=16, stride=16 // 2):
        super(DeepDecoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.PReLU(),
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.PReLU(),
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.PReLU(),
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose1d(N, 1, kernel_size=kernel_size, stride=stride, bias=True)
        )

    def forward(self, x):
        """
        x: N x L or N x C x L
        """
        x = self.sequential(x)
        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)

        return x
