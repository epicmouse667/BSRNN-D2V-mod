from typing import Tuple, List
import torch
import torch.nn as nn

from wesep.modules.dpccn.convs import Conv2dBlock

class DenseUNet(nn.Module):
    def __init__(self,
                 win_len: int = 512,  # 32 ms
                 win_inc: int = 128,  # 8 ms
                 fft_len: int = 512,
                 win_type: str = "sqrthann",
                 kernel_size: Tuple[int] = (3, 3),
                 stride1: Tuple[int] = (1, 1),
                 stride2: Tuple[int] = (1, 2),
                 paddings: Tuple[int] = (1, 0),
                 output_padding: Tuple[int] = (0, 0),
                 tcn_dims: int = 384,
                 tcn_blocks: int = 10,
                 tcn_layers: int = 2,
                 causal: bool = False,
                 pool_size: Tuple[int] = (4, 8, 16, 32),
                 num_spks: int = 1,
                 L: int = 20) -> None:
        super(DenseUNet, self).__init__()

        self.L = L
        self.fft_len = fft_len
        self.num_spks = num_spks

        self.stft = ConvSTFT(win_len, win_inc, fft_len, win_type)
        self.conv2d = nn.Conv2d(2, 16, kernel_size, stride1, paddings)

        freq_bins = fft_len // 2 + 1
        self.speaker_enc = SpeakerEnc(freq_bins, freq_bins - 2)

        self.encoder = self._build_encoder(
            kernel_size=kernel_size,
            stride=stride2,
            padding=paddings
        )
        self.tcn_layers = self._build_tcn_layers(
            tcn_layers,
            tcn_blocks,
            in_dims=tcn_dims,
            out_dims=tcn_dims,
            causal=causal
        )
        self.decoder = self._build_decoder(
            kernel_size=kernel_size,
            stride=stride2,
            padding=paddings,
            output_padding=output_padding
        )
        self.avg_pool = self._build_avg_pool(pool_size)
        self.avg_proj = nn.Conv2d(64, 32, 1, 1)

        self.deconv2d = nn.ConvTranspose2d(32, 2 * num_spks, kernel_size, stride1, paddings)
        self.istft = ConviSTFT(win_len, win_inc, fft_len, win_type, 'complex')

    def _build_encoder(self, **enc_kargs):
        """
        Build encoder layers 
        """
        encoder = nn.ModuleList()
        encoder.append(DenseBlock(16, 16, "enc"))
        for i in range(4):
            encoder.append(
                nn.Sequential(
                    Conv2dBlock(in_dims=16 if i == 0 else 32,
                                out_dims=32, **enc_kargs),
                    DenseBlock(32, 32, "enc")
                )
            )
        encoder.append(Conv2dBlock(in_dims=32, out_dims=64, **enc_kargs))
        encoder.append(Conv2dBlock(in_dims=64, out_dims=128, **enc_kargs))
        encoder.append(Conv2dBlock(in_dims=128, out_dims=384, **enc_kargs))

        return encoder

    def _build_decoder(self, **dec_kargs):
        """
        Build decoder layers 
        """
        decoder = nn.ModuleList()
        decoder.append(ConvTrans2dBlock(in_dims=384 * 2, out_dims=128, **dec_kargs))
        decoder.append(ConvTrans2dBlock(in_dims=128 * 2, out_dims=64, **dec_kargs))
        decoder.append(ConvTrans2dBlock(in_dims=64 * 2, out_dims=32, **dec_kargs))
        for i in range(4):
            decoder.append(
                nn.Sequential(
                    DenseBlock(32, 64, "dec"),
                    ConvTrans2dBlock(in_dims=64,
                                     out_dims=32 if i != 3 else 16,
                                     **dec_kargs)
                )
            )
        decoder.append(DenseBlock(16, 32, "dec"))

        return decoder

    def _build_tcn_blocks(self, tcn_blocks, **tcn_kargs):
        """
        Build TCN blocks in each repeat (layer)
        """
        blocks = [
            TCNBlock(**tcn_kargs, dilation=(2 ** b))
            for b in range(tcn_blocks)
        ]

        return nn.Sequential(*blocks)

    def _build_tcn_layers(self, tcn_layers, tcn_blocks, **tcn_kargs):
        """
        Build TCN layers
        """
        layers = [
            self._build_tcn_blocks(tcn_blocks, **tcn_kargs)
            for _ in range(tcn_layers)
        ]

        return nn.Sequential(*layers)

    def _build_avg_pool(self, pool_size):
        """
        Build avg pooling layers
        """
        avg_pool = nn.ModuleList()
        for sz in pool_size:
            avg_pool.append(
                nn.Sequential(
                    nn.AvgPool2d(sz),
                    nn.Conv2d(32, 8, 1, 1)
                )
            )

        return avg_pool

    def wav2spec(self, x: torch.Tensor, mags: bool = False) -> torch.Tensor:
        """
        convert waveform to spectrogram
        """
        assert x.dim() == 2
        x = x / torch.std(x, -1, keepdims=True)  # variance normalization
        specs = self.stft(x)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        spec = torch.stack([real, imag], 1)
        spec = torch.einsum("hijk->hikj", spec)  # batchsize, 2, T, F
        if mags:
            return torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        else:
            return spec

    def sep(self, spec: torch.Tensor) -> List[torch.Tensor]:
        """
        spec: (batchsize, 2, T, F)
        return [real, imag] or waveform
        """
        spec = torch.einsum("hijk->hikj", spec)  # (batchsize, 2, F, T)
        B, N, F, T = spec.shape
        est = torch.chunk(spec, 2, 1)  # [(B, 1, F, T), (B, 1, F, T)]
        est = torch.cat(est, 2).reshape(B, -1, T)  # B, 2F, T
        return torch.squeeze(self.istft(est))

    def forward(self,
                mix: torch.Tensor,
                aux: torch.Tensor) -> torch.Tensor:
        """
        if waveform = True, return both waveform and real & imag parts;
        else, only return real & imag parts
        """
        if mix.dim() == 1:
            mix = torch.unsqueeze(mix, 0)
            aux = torch.unsqueeze(aux, 0)

        # speaker encoder
        aux = self.wav2spec(aux, True)
        aux = self.speaker_enc(aux)

        # speech separation
        mix_spec = self.wav2spec(mix, False)
        out = self.conv2d(mix_spec)
        out_list = []
        out = self.encoder[0](out)
        out = out * aux  # SA

        out_list.append(out)

        for _, enc in enumerate(self.encoder[1:]):
            out = enc(out)
            out_list.append(out)

        B, N, T, F = out.shape
        out = out.reshape(B, N, T * F)
        out = self.tcn_layers(out)
        out = out.reshape(B, N, T, F)

        out_list = out_list[::-1]
        for idx, dec in enumerate(self.decoder):
            out = dec(torch.cat([out_list[idx], out], 1))

            # Pyramidal pooling
        B, N, T, F = out.shape
        upsample = nn.Upsample(size=(T, F), mode='bilinear')
        pool_list = []
        for avg in self.avg_pool:
            pool_list.append(upsample(avg(out)))
        out = torch.cat([out, *pool_list], 1)
        out = self.avg_proj(out)

        out = self.deconv2d(out)

        return self.sep(out)