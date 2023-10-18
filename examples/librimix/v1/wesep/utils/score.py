import numpy as np
from pesq import pesq
from pystoi.stoi import stoi
from torchmetrics.text import WordErrorRate 

def cal_SISNR(est, ref, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref)
    est_zm = est - np.mean(est)
    ref_zm = ref - np.mean(ref)

    t = np.sum(est_zm * ref_zm) * ref_zm / (np.linalg.norm(ref_zm) ** 2 + eps)
    return 20 * np.log10(eps + np.linalg.norm(t) / (np.linalg.norm(est_zm - t) + eps))


def cal_SISNRi(est, ref, mix, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    sisnr1 = cal_SISNR(est, ref)
    sisnr2 = cal_SISNR(mix, ref)

    return sisnr1, sisnr1 - sisnr2

def cal_WER(est,ref):
    assert isinstance(est,str) and isinstance(ref,str)
    est = est.upper()
    ref = ref.upper()
    wer = WordErrorRate()
    return wer([est],[ref])

def cal_PESQ(est, ref):
    assert len(est) == len(ref)
    mode = 'wb'
    p = pesq(16000, ref, est, mode)
    return p


def cal_PESQi(est, ref, mix):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    pesq1 = cal_PESQ(est, ref)
    pesq2 = cal_PESQ(mix, ref)

    return pesq1, pesq1 - pesq2


def cal_STOI(est, ref):
    assert len(est) == len(ref)
    p = stoi(ref, est, 16000)
    return p


def cal_STOIi(est, ref, mix):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    stoi1 = cal_STOI(est, ref)
    stoi2 = cal_STOI(mix, ref)

    return stoi1, stoi1 - stoi2