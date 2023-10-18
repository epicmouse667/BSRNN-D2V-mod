import logging
import math
import torch.nn as nn
import torch.nn.functional as F

import fast_bss_eval
import torch
from abc import ABC
from wesep.utils.abs_loss import AbsEnhLoss

EPS = 1e-6

def cal_SISNR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]

    Code from https://github.com/zexupan/avse_hybrid_loss/blob/main/src/av-dprnn/utils.py#L171
    """
    assert source.size() == estimate_source.size()

    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis=-1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis=-1,
                                                   keepdim=True)

    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis=-1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis=-1,
                     keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis=-1) / (
            torch.sum(noise ** 2, axis=-1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)

    return sisnr


class SISNRLoss(torch.nn.Module):
    """SI-SNR (or named SI-SDR) loss

    A more stable SI-SNR loss with clamp from `fast_bss_eval`.

    Attributes:
        clamp_db: float
            clamp the output value in  [-clamp_db, clamp_db]
        zero_mean: bool
            When set to True, the mean of all signals is subtracted prior.
        eps: float
            Deprecated. Kept for compatibility.
    """

    def __init__(
            self,
            clamp_db=None,
            zero_mean=True,
            eps=None,
    ):
        super().__init__()

        self.clamp_db = clamp_db
        self.zero_mean = zero_mean
        if eps is not None:
            logging.warning(
                "Eps is deprecated in si_snr loss, set clamp_db instead.")
            if self.clamp_db is None:
                self.clamp_db = -math.log10(eps / (1 - eps)) * 10

    def SI_SNR(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        """SI-SNR forward.

        Args:

            ref: Tensor, (..., n_samples)
                reference signal
            est: Tensor (..., n_samples)
                estimated signal

        Returns:
            loss: (...,)
                the SI-SDR loss (negative si-sdr)
        """
        assert torch.is_tensor(est) and torch.is_tensor(ref), est

        si_snr = fast_bss_eval.si_sdr_loss(
            est=est,
            ref=ref,
            zero_mean=self.zero_mean,
            clamp_db=self.clamp_db,
            pairwise=False,
        )

        return si_snr

    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        si_snr = self.SI_SNR(ref, est)
        loss = -si_snr
        return loss

class TimeDomainLoss(AbsEnhLoss, ABC):
    """Base class for all time-domain Enhancement loss modules."""

    @property
    def name(self) -> str:
        return self._name

    @property
    def only_for_test(self) -> bool:
        return self._only_for_test

    @property
    def is_noise_loss(self) -> bool:
        return self._is_noise_loss

    @property
    def is_dereverb_loss(self) -> bool:
        return self._is_dereverb_loss

    def __init__(
        self,
        name,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        super().__init__()
        # only used during validation
        self._only_for_test = only_for_test
        # only used to calculate the noise-related loss
        self._is_noise_loss = is_noise_loss
        # only used to calculate the dereverberation-related loss
        self._is_dereverb_loss = is_dereverb_loss
        if is_noise_loss and is_dereverb_loss:
            raise ValueError(
                "`is_noise_loss` and `is_dereverb_loss` cannot be True at the same time"
            )
        if is_noise_loss and "noise" not in name:
            name = name + "_noise"
        if is_dereverb_loss and "dereverb" not in name:
            name = name + "_dereverb"
        self._name = name


EPS = torch.finfo(torch.get_default_dtype()).eps

class SNRLoss(TimeDomainLoss):
    def __init__(
        self,
        eps=EPS,
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "snr_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self.eps = float(eps)

    def forward(self, ref: torch.Tensor, inf: torch.Tensor) -> torch.Tensor:
        # the return tensor should be shape of (batch,)

        noise = inf - ref

        snr = 20 * (
            torch.log10(torch.norm(ref, p=2, dim=1).clamp(min=self.eps))
            - torch.log10(torch.norm(noise, p=2, dim=1).clamp(min=self.eps))
        )
        return -snr

class SISNRLoss(TimeDomainLoss):
    """SI-SNR (or named SI-SDR) loss

    A more stable SI-SNR loss with clamp from `fast_bss_eval`.

    Attributes:
        clamp_db: float
            clamp the output value in  [-clamp_db, clamp_db]
        zero_mean: bool
            When set to True, the mean of all signals is subtracted prior.
        eps: float
            Deprecated. Kept for compatibility.
    """

    def __init__(
        self,
        clamp_db=None,
        zero_mean=True,
        eps=None,
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "si_snr_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self.clamp_db = clamp_db
        self.zero_mean = zero_mean
        if eps is not None:
            logging.warning("Eps is deprecated in si_snr loss, set clamp_db instead.")
            if self.clamp_db is None:
                self.clamp_db = -math.log10(eps / (1 - eps)) * 10

    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        """SI-SNR forward.

        Args:

            ref: Tensor, (..., n_samples)
                reference signal
            est: Tensor (..., n_samples)
                estimated signal

        Returns:
            loss: (...,)
                the SI-SDR loss (negative si-sdr)
        """
        assert torch.is_tensor(est) and torch.is_tensor(ref), est

        si_snr = fast_bss_eval.si_sdr_loss(
            est=est,
            ref=ref,
            zero_mean=self.zero_mean,
            clamp_db=self.clamp_db,
            pairwise=False,
        )

        return si_snr
    


class SISNR_CTC_Loss(TimeDomainLoss):
    def __init__(
        self,
        alpha,
        name = None,
        zero_mean=True,
        clamp_db=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "si_snr_ctc_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )
        self.alpha = alpha
        self.zero_mean = zero_mean
        self.clamp_db = clamp_db

    def PadAware_SISNR_Wrapper(
        self,
        est_wav,
        target_wav,
        wav_len
    ):
        loss = []
        batch_size = est_wav.shape[0]
        assert est_wav.shape[0] == target_wav.shape[0] == wav_len.shape[0]
        for i in range(batch_size):
            est = est_wav[i,:wav_len[i]]
            ref = target_wav[i,:wav_len[i]]
            assert est.shape == ref.shape,'est shape: {} ref shape: {}'.format(est.shape,ref.shape)
            si_snr = fast_bss_eval.si_sdr_loss(
                est=est,
                ref=ref,
                zero_mean=self.zero_mean,
                clamp_db=self.clamp_db,
                pairwise=False,
            )
            loss.append(si_snr)
        return torch.tensor(loss)   
    def forward(
        self,
        est_wav,
        target_wav,
        wav_len,
        log_prob,
        prob_len,
        label,
        label_len,
    ):
        log_prob = log_prob.transpose(0,1)
        batch_size = prob_len.shape[0]
        assert label.shape[0] == label_len.shape[0] ==prob_len.shape[0] == log_prob.shape[1]
        assert label.max() <= log_prob.shape[-1]
        asr_loss =  F.ctc_loss(
                    log_prob,
                    label,
                    prob_len,
                    label_len,
                    reduction='none',
        ).cuda()
        assert asr_loss.shape[0] == batch_size
        # si_snr = self.PadAware_SISNR_Wrapper(
        #     est_wav,
        #     target_wav,
        #     wav_len,
        # ).cuda()
        si_snr = fast_bss_eval.si_sdr_loss(
                est=est_wav,
                ref=target_wav,
                zero_mean=True,
                clamp_db=None,
                pairwise=False,
            )
        loss = asr_loss * self.alpha + si_snr
        return loss,si_snr,asr_loss