"""!
@author Yi Luo (oulyluo)
@copyright Tencent AI Lab
Modified by Shuai Wang (wsstriving@gmail.com)
"""

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn


class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size, use_bidirectional):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps

        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=use_bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * (int(use_bidirectional) + 1), input_size)

    def forward(self, input):
        # input shape: batch, dim, seq

        rnn_output, _ = self.rnn(self.norm(input).transpose(1, 2).contiguous())
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(input.shape[0],
                                                                                           input.shape[2],
                                                                                           input.shape[1])

        return input + rnn_output.transpose(1, 2).contiguous()


class BSNet(nn.Module):
    def __init__(self, in_channel, n_band=7, n_layers=1, use_bidirectional=True):
        super(BSNet, self).__init__()

        self.n_band = n_band
        self.feature_dim = in_channel // n_band

        band_rnn_layers = nn.ModuleList()

        for _ in range(n_layers):
            band_rnn_layers.append(ResRNN(self.feature_dim, self.feature_dim * 2, use_bidirectional=use_bidirectional))
        self.band_rnn = nn.Sequential(band_rnn_layers)
        self.band_comm = ResRNN(self.feature_dim, self.feature_dim * 2, use_bidirectional=use_bidirectional)

    def forward(self, input):
        # input shape: B, n_band * N, T
        B, N, T = input.shape

        band_output = self.band_rnn(input.view(B * self.n_band, self.feature_dim, -1)).view(B, self.n_band, -1, T)

        # band comm
        band_output = band_output.permute(0, 3, 2, 1).contiguous().view(B * T, -1, self.n_band)
        output = self.band_comm(band_output).view(B, T, -1, self.n_band).permute(0, 3, 2, 1).contiguous()

        return output.view(B, N, T)


class Separator(nn.Module):
    def __init__(self, sr=16000, win=1024, stride=160, feature_dim=80, num_repeat=6, use_bidirectional=True):
        super(Separator, self).__init__()

        self.sr = sr
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps

        # 0-1k (100 hop), 1k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf
        bandwidth_50 = int(np.floor(50 / (sr / 2.) * self.enc_dim))
        bandwidth_100 = int(np.floor(100 / (sr / 2.) * self.enc_dim))
        bandwidth_250 = int(np.floor(250 / (sr / 2.) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.) * self.enc_dim))
        # bandwidth_1k = int(np.floor(1000 / (sr / 2.) * self.enc_dim))
        # bandwidth_2k = int(np.floor(2000 / (sr / 2.) * self.enc_dim))

        self.band_width = [bandwidth_100] * 10
        self.band_width += [bandwidth_250] * 12
        self.band_width += [bandwidth_500] * 8

        self.band_width.append(self.enc_dim - np.sum(self.band_width))

        self.n_band = len(self.band_width)
        # print(self.band_width)

        self.BN = nn.ModuleList([])
        for i in range(self.n_band):
            self.BN.append(nn.Sequential(nn.GroupNorm(1, self.band_width[i] * 2, self.eps),
                                         nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1)
                                         )
                           )

        self.separator = []
        for i in range(num_repeat):
            self.separator.append(BSNet(self.n_band * self.feature_dim, self.n_band))
        self.separator = nn.Sequential(*self.separator)

        self.mask = nn.ModuleList([])
        for i in range(self.n_band):
            self.mask.append(nn.Sequential(nn.GroupNorm(1, self.feature_dim, torch.finfo(torch.float32).eps),
                                           nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                                           nn.Tanh(),
                                           nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                                           nn.Tanh(),
                                           nn.Conv1d(self.feature_dim * 4, self.band_width[i] * 4, 1)
                                           )
                             )

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input):
        # input shape: (B, C, T)

        batch_size, nch, nsample = input.shape
        input = input.view(batch_size * nch, -1)

        # frequency-domain separation
        spec = torch.stft(input, n_fft=self.win, hop_length=self.stride,
                          window=torch.hann_window(self.win).to(input.device).type(input.type()),
                          return_complex=True)

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(spec_RI[:, :, band_idx:band_idx + self.band_width[i]].contiguous())
            subband_mix_spec.append(spec[:, band_idx:band_idx + self.band_width[i]])  # B*nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(self.BN[i](subband_spec[i].view(batch_size * nch, self.band_width[i] * 2, -1)))
        subband_feature = torch.stack(subband_feature, 1)  # B, n_band, N, T

        # import pdb; pdb.set_trace()
        # separator
        sep_output = self.separator(
            subband_feature.view(batch_size * nch, self.n_band * self.feature_dim, -1))  # B, n_band*N, T
        sep_output = sep_output.view(batch_size * nch, self.n_band, self.feature_dim, -1)

        sep_subband_spec = []
        for i in range(len(self.band_width)):
            this_output = self.mask[i](sep_output[:, i]).view(batch_size * nch, 2, 2, self.band_width[i], -1)
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:, 0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:, 1]  # B*nch, K, BW, T
            est_spec_real = subband_mix_spec[i].real * this_mask_real - subband_mix_spec[
                i].imag * this_mask_imag  # B*nch, BW, T
            est_spec_imag = subband_mix_spec[i].real * this_mask_imag + subband_mix_spec[
                i].imag * this_mask_real  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T

        output = torch.istft(est_spec.view(batch_size * nch, self.enc_dim, -1),
                             n_fft=self.win, hop_length=self.stride,
                             window=torch.hann_window(self.win).to(input.device).type(input.type()), length=nsample)

        output = output.view(batch_size, nch, -1)

        return output


if __name__ == '__main__':
    from thop import profile, clever_format

    model = Separator(sr=44100, win=2048, stride=512, feature_dim=80, num_repeat=10)
    s = 0
    for param in model.parameters():
        s += np.product(param.size())
    print('# of parameters: ' + str(s / 1024.0 / 1024.0))

    x = torch.randn((1, 2, 44100 * 3))
    output = model(x)
    print(output.shape)

    macs, params = profile(model, inputs=(x,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
