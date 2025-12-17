import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "Conv",
    "APSM"
)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

class APSM(nn.Module):
    def __init__(self, c1, c2):
        super(APSM, self).__init__()
        assert c1 == c2
        self.rgb2gray = transforms.Grayscale(num_output_channels=1)
        self.cv2 = Conv(1, 3, 1, 1)
        self.amplitude_spectrum_path = "apsm_works/apsm_amplitude/plane_priori_amplitude_normalized.npy"
        # self.amplitude_spectrum_path = "apsm_works/apsm_amplitude/ship_priori_amplitude_normalized.npy"
        self.magnitude = torch.Tensor(np.load(self.amplitude_spectrum_path)).view(1, 1, 640, 640)
        self.magnitude_eval = None

    def forward(self, x):
        x = self.rgb2gray(x)
        x_patch_fft = torch.fft.fft2(x.float())
        x_patch_fft_shift = torch.fft.fftshift(x_patch_fft)
        x_pha = x_patch_fft_shift.angle()
        if self.magnitude.device != x.device:
            self.magnitude = self.magnitude.to(x.device)
        if self.train() and (x_pha.shape[2], x_pha.shape[3]) == self.magnitude.squeeze().shape:
            x_patch_fft = self.magnitude * torch.exp(1j * x_pha)
            self.magnitude_eval = None
        else:
            if self.magnitude_eval is None or self.magnitude_eval.squeeze().shape != (x_pha.shape[2], x_pha.shape[3]):
                magnitude_avg = torch.Tensor(np.load(self.amplitude_spectrum_path)).view(1, 1, 640, 640)
                self.magnitude_eval = torch.nn.functional.pad(magnitude_avg,
                                                              [int((x_pha.shape[2] - magnitude_avg.shape[2]) / 2),
                                                               int((x_pha.shape[2] - magnitude_avg.shape[2]) / 2),
                                                               int((x_pha.shape[3] - magnitude_avg.shape[3]) / 2),
                                                               int((x_pha.shape[3] - magnitude_avg.shape[3]) / 2)],
                                                              "constant", 0)
            if self.magnitude_eval.device != x_pha.device:
                self.magnitude_eval = self.magnitude_eval.to(x_pha.device)
            x_patch_fft = self.magnitude_eval * torch.exp(1j * x_pha)
        # x_patch = torch.fft.ifft2(torch.fft.ifftshift(x_patch_fft), dim=(-2, -1)).abs().to(dtype=x.dtype)
        x_patch = torch.fft.ifft2(torch.fft.ifftshift(x_patch_fft), dim=(-2, -1)).real.to(dtype=x.dtype)
        return self.cv2(x_patch)
