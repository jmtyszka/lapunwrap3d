"""
ChatGPT 4o port of original Matlab code from the qMRLab package to python 3.10
- Refactored, corrected and optimized as a python class by JMT
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftshift


class LaplacianPhaseUnwrap3D():

    def __init__(self, phi_w):

        self.phi_w = phi_w
        self.phi_uw = np.zeros_like(phi_w)
        self.phi_uw_corr = np.zeros_like(phi_w)

    def unwrap(self):

        """
        Fourier Laplacian phase unwrapping in 3D

        - Initially ported from the qMRLab Matlab function unwrapPhaseLaplacian.m (ChatGTP 4o)
        - Corrected, refactored and optimized for python (JMT)
        """

        # Get the size of the wrapped phase input
        nx, ny, nz = self.phi_w.shape
        hx, hy, hz = nx // 2, ny // 2, nz // 2

        # Kernel widths and half widths
        kx, ky, kz = 3, 3, 3
        hkx, hky, hkz = (kx - 1)//2, (ky - 1)//2, (kz - 1)//2

        # Discrete Laplacian operator kernel
        lap_kernel = np.array([
                [[ 0,  0,  0], [ 0, -1,  0], [0,  0, 0]],
                [[ 0, -1,  0], [-1,  6, -1], [0, -1, 0]],
                [[ 0,  0,  0], [ 0, -1,  0], [0,  0, 0]],
        ])

        # Center and zero-pad kernel to image size
        lap = np.zeros([nx, ny, nz])
        lap[hx-hkx:hx+hkx+1, hy-hky:hy+hky+1, hz-hkz:hz+hkz+1] = lap_kernel

        # FFT of the zero-padded Laplacian kernel
        # Note that if the centered Laplacian kernel is fftshifted initially
        # there's no need to fftshift anywhere else. Stick low frequency at edges/corners of k-space
        fft_lap = fftn(fftshift(lap))

        # Reciprocal of Laplacian FFT
        # Mask zeros for division
        fft_lap_recip = np.zeros_like(fft_lap)
        nonzero = fft_lap.nonzero()
        fft_lap_recip[nonzero] = 1.0 / fft_lap[nonzero]

        # Compute the Laplacian of the phase by Fourier convolution
        sin_phi = np.sin(self.phi_w)
        cos_phi = np.cos(self.phi_w)
        fft_lap_phase = (
            cos_phi * ifftn(fftn(sin_phi) * fft_lap) -
            sin_phi * ifftn(fftn(cos_phi) * fft_lap)
        )

        # Laplacian unwrapped 3D phase image
        self.phi_uw = np.real(ifftn(fftn(fft_lap_phase) * fft_lap_recip))

        return self.phi_uw