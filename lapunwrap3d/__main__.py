import nibabel as nib
import argparse
from .unwrap import LaplacianPhaseUnwrap3D


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Laplacian phase unwrap in 3D')
    parser.add_argument('-i', '--infile', required=True, help='Nifti phase image in range -pi to pi')

    args = parser.parse_args()
    phi_w_fname = args.infile

    # Remove up to two extensions ('.nii' or '.nii.gz')
    phi_w_fstub = phi_w_fname.rsplit('.', 1)[0].rsplit('.', 1)[0]

    phi_uw_fname = f"{phi_w_fstub}_lapuw.nii.gz"
    phi_uw_corr_fname = f"{phi_w_fstub}_lapuw_corr.nii.gz"

    # Load phase image
    print('Loading wrapped phase image')
    phi_w_nii = nib.load(phi_w_fname)
    phi_w = phi_w_nii.get_fdata()

    print('Unwrapping phase')
    lpu = LaplacianPhaseUnwrap3D(phi_w)
    phi_uw = lpu.unwrap()

    print(f'Saving unwrapped phase image to {phi_uw_fname}')
    phi_uw_nii = nib.Nifti1Image(phi_uw, affine=phi_w_nii.affine)
    phi_uw_nii.to_filename(phi_uw_fname)


if __name__ == '__main__':

    main()