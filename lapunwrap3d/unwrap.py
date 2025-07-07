"""
Laplacian phase unwrapping in 3D

AUTHOR : Mike Tyszka
PLACE  : Caltech Brain Imaging Center
DATES  : 2024-10-13 JMT Implement directly from Schofield and Zhu 2003
         2024-10-14 JMT Skip iterative phase difference correct - doesn't work
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from numpy.fft import (fftn, ifftn, fftshift)


class LaplacianPhaseUnwrap3D():

    def __init__(self, phi_w):

        # Check dimensionality of phi_w
        ndims = phi_w.ndim
        if ndims < 3 or ndims > 4:
            raise ValueError('LaplacianPhaseUnwrap3D requires 3D or 3D x t phase images')
        self._ndims = ndims

        self.phi_w = phi_w
        self.phi_uw = np.zeros_like(phi_w)

        # Useful image dimensions
        nx, ny, nz = self.phi_w.shape[0:3]
        hx, hy, hz = nx // 2, ny // 2, nz // 2

        # k-space coordinate meshes
        # Use 'ij' indexing to preserve dimension order in output meshes
        kx = np.arange(-hx, hx) / nx
        ky = np.arange(-hy, hy) / ny
        kz = np.arange(-hz, hz) / nz
        kxm, kym, kzm = np.meshgrid(kx, ky, kz, indexing='ij')

        # Precalculate Laplacian operator and its inverse in k-space
        self._lap_k = fftshift(kxm ** 2 + kym ** 2 + kzm ** 2)
        self._invlap_k = np.reciprocal(self._lap_k, out=np.zeros_like(self._lap_k), where=self._lap_k != 0)

    def unwrap(self):

        phi_uw = np.zeros_like(self.phi_w)

        if self._ndims == 3:
            phi_uw = self._unwrap3d(self.phi_w)

        if self._ndims == 4:
            for tc in range(self.phi_w.shape[-1]):
                phi_uw[..., tc] = self._unwrap3d(self.phi_w[..., tc])

        self.phi_uw = phi_uw

        return phi_uw

    def _unwrap3d(self, phi_w):
        """
        Fourier Laplacian phase unwrapping in 3D
        - Direct implementation of Schofield and Zhu 2003 algorithm
        """

        # Pre-calculate sine and cosine of wrapped phase image
        cos_phi_w = np.cos(phi_w)
        sin_phi_w = np.sin(phi_w)

        phi_uw = np.real(
            self._invlap(
                cos_phi_w * self._lap(sin_phi_w) -
                sin_phi_w * self._lap(cos_phi_w),
            )
        )

        return phi_uw

    def _lap(self, x):
        return ifftn(fftn(x) * self._lap_k)

    def _invlap(self, x):
        return ifftn(fftn(x) * self._invlap_k)


