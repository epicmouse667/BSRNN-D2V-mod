import torch.nn as nn

from wesep.modules.tasnet.convs import Conv1DBlock


class Separation(nn.Module):
    def __init__(self, R, X, B, H, P, norm='gLN', causal=False, skip_con=True):
        """

        :param R: Number of repeats
        :param X: Number of convolutional blocks in each repeat
        :param B: Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
        :param H: Number of channels in convolutional blocks
        :param P: Kernel size in convolutional blocks
        :param norm: The type of normalization(gln, cln, bn)
        :param causal: Two choice(causal or noncausal)
        :param skip_con: Whether to use skip connection
        """
        super(Separation, self).__init__()
        self.separation = nn.ModuleList([])
        for r in range(R):
            for x in range(X):
                self.separation.append(Conv1DBlock(
                    B, H, P, 2 ** x, norm, causal, skip_con))
        self.skip_con = skip_con

    def forward(self, x):
        '''
           x: [B, N, L]
           out: [B, N, L]
        '''
        if self.skip_con:
            skip_connection = 0
            for i in range(len(self.separation)):
                skip, out = self.separation[i](x)
                skip_connection = skip_connection + skip
                x = out
            return skip_connection
        else:
            for i in range(len(self.separation)):
                out = self.separation[i](x)
                x = out
            return x
