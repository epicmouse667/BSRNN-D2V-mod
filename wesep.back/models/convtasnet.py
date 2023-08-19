import torch
import torch.nn as nn

from wesep.modules.common import select_norm
from wesep.modules.tasnet import DeepDecoder
from wesep.modules.tasnet import DeepEncoder
from wesep.modules.tasnet import Separation
from wesep.modules.tasnet.convs import Conv1D
from wesep.modules.tasnet.convs import ConvTrans1D
from wesep.modules.common.spkadapt import SpeakerTransform, SpeakerFuseLayer

class ConvTasNet(nn.Module):
    def __init__(self,
                 N=512,
                 L=16,
                 B=128,
                 H=512,
                 P=3,
                 X=8,
                 R=3,
                 norm="gLN",
                 activate="relu",
                 causal=False,
                 skip_con=False,
                 use_deep_enc=False,
                 use_deep_dec=True):
        """
        :param N: Number of filters in autoencoder
        :param L: Length of the filters (in samples)
        :param B: Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
        :param H: Number of channels in convolutional blocks
        :param P: Kernel size in convolutional blocks
        :param X: Number of convolutional blocks in each repeat
        :param R: Number of repeats
        :param norm:
        :param activate:
        :param causal:
        :param skip_con:
        :param use_deep_enc:
        :param use_deep_dec:
        """
        super(ConvTasNet, self).__init__()

        # n x 1 x T => n x N x T
        if use_deep_enc:
            self.encoder = DeepEncoder(1, N, L, stride=L // 2)
        else:
            self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)
        self.spk_transform = SpeakerTransform()
        self.spk_fuse = SpeakerFuseLayer(fuse_type='concat')
        # n x N x T  Layer Normalization of Separation
        self.LayerN_S = select_norm('cLN', N)
        # n x B x T  Conv 1 x 1 of  Separation
        self.BottleN_S = Conv1D(N, B, 1)
        # Separation block
        # n x B x T => n x B x T
        self.separation = Separation(R, X, B, H, P, norm=norm, causal=causal, skip_con=skip_con)
        # n x B x T => n x 2*N x T
        self.gen_masks = Conv1D(B, N, 1)
        # n x N x T => n x 1 x L
        if use_deep_dec:
            self.decoder = DeepDecoder(N, L, stride=L // 2)
        else:
            self.decoder = ConvTrans1D(N, 1, L, stride=L // 2)
        # activation function
        active_f = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=0)
        }
        self.activation_type = activate
        self.activation = active_f[activate]

    def forward(self, x, embeddings):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        # x: n x 1 x L => n x N x T
        w = self.encoder(x)
        spk_embeds = embeddings.unsqueeze(-1)
        spk_embeds = self.spk_transform(spk_embeds)
        w_1 = self.spk_fuse(w, spk_embeds)
        # n x N x L => n x B x L
        e = self.LayerN_S(w_1)
        e = self.BottleN_S(e)
        # n x B x L => n x B x L
        e = self.separation(e)
        # n x B x L => n x N x L
        m = self.gen_masks(e)
        # n x N x L
        # n x N x L
        m = self.activation(m)
        d = w * m
        # decoder part  n x L
        s = self.decoder(d)
        return s


def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


def test_convtasnet():
    x = torch.randn(4, 32000)
    net = ConvTasNet()
    s = net(x)
    print(str(check_parameters(net)) + ' Mb')
    print(s[1].shape)


if __name__ == "__main__":
    test_convtasnet()
