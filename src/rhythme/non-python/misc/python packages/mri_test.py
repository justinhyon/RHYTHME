import os.path as op
import nibabel
from nilearn.plotting import plot_glass_brain
import numpy as np
import time
from timeit import default_timer as timer
import re

import mne
from mne.channels import compute_native_head_t, read_custom_montage, make_standard_montage, make_dig_montage
from mne.viz import plot_alignment
from mne.forward import write_forward_solution, read_forward_solution
from mayavi import mlab
import matplotlib.pyplot as plt
from mne.bem import make_watershed_bem, write_bem_solution, read_bem_solution
from mne.source_space import get_volume_labels_from_aseg, get_volume_labels_from_src
from nilearn import plotting

mgz_fname='/Users/jasonhyon/Desktop/Exp2-170719/SS-FLOW-04_MD/freesurfer/MD/MD/mri/T1.mgz'
subjects_dir = '/Users/jasonhyon/Desktop/Exp2-170719/SS-FLOW-04_MD/freesurfer/MD'
subject = 'MD'


t1 = nibabel.load(mgz_fname)
t1.orthoview()
data = np.asarray(t1.dataobj)
print(data.shape)

print(t1.affine)
vox = np.array([122, 119, 102])
xyz_ras = apply_trans(t1.affine, vox)
print('Our voxel has real-world coordinates {}, {}, {} (mm)'
      .format(*np.round(xyz_ras, 3)))
ras_coords_mm = np.array([1, -17, -18])
inv_affine = np.linalg.inv(t1.affine)
i_, j_, k_ = np.round(apply_trans(inv_affine, ras_coords_mm)).astype(int)
print('Our real-world coordinates correspond to voxel ({}, {}, {})'
      .format(i_, j_, k_))

def imshow_mri(data, img, vox, xyz, suptitle):
    """Show an MRI slice with a voxel annotated."""
    i, j, k = vox
    fig, ax = plt.subplots(1, figsize=(6, 6))
    codes = nibabel.orientations.aff2axcodes(img.affine)
    # Figure out the title based on the code of this axis
    ori_slice = dict(P='Coronal', A='Coronal',
                     I='Axial', S='Axial',
                     L='Sagittal', R='Saggital')
    ori_names = dict(P='posterior', A='anterior',
                     I='inferior', S='superior',
                     L='left', R='right')
    title = ori_slice[codes[0]]
    ax.imshow(data[i], vmin=10, vmax=120, cmap='gray', origin='lower')
    ax.axvline(k, color='y')
    ax.axhline(j, color='y')
    for kind, coords in xyz.items():
        annotation = ('{}: {}, {}, {} mm'
                      .format(kind, *np.round(coords).astype(int)))
        text = ax.text(k, j, annotation, va='baseline', ha='right',
                       color=(1, 1, 0.7))
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()])
    # reorient view so that RAS is always rightward and upward
    x_order = -1 if codes[2] in 'LIP' else 1
    y_order = -1 if codes[1] in 'LIP' else 1
    ax.set(xlim=[0, data.shape[2] - 1][::x_order],
           ylim=[0, data.shape[1] - 1][::y_order],
           xlabel=f'k ({ori_names[codes[2]]}+)',
           ylabel=f'j ({ori_names[codes[1]]}+)',
           title=f'{title} view: i={i} ({ori_names[codes[0]]}+)')
    fig.suptitle(suptitle)
    fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)
    return fig


imshow_mri(data, t1, vox, {'Scanner RAS': xyz_ras}, 'MRI slice')

Torig = t1.header.get_vox2ras_tkr()
print(t1.affine)
print(Torig)
xyz_mri = apply_trans(Torig, vox)
imshow_mri(data, t1, vox, dict(MRI=xyz_mri), 'MRI slice')

fiducials = mne.coreg.get_mni_fiducials(subject, subjects_dir=subjects_dir)
nasion_mri = [d for d in fiducials if d['ident'] == FIFF.FIFFV_POINT_NASION][0]
print(nasion_mri)  # note it's in Freesurfer MRI coords
nasion_mri = nasion_mri['r'] * 1000  # meters → millimeters
nasion_vox = np.round(
    apply_trans(np.linalg.inv(Torig), nasion_mri)).astype(int)
imshow_mri(data, t1, nasion_vox, dict(MRI=nasion_mri),
           'Nasion estimated from MRI transform')

rr_mm, tris = mne.read_surface('/Users/jasonhyon/Desktop/Exp2-170719/SS-FLOW-04_MD/freesurfer/MD/MD/surf/rh.white')
print(f'rr_mm.shape == {rr_mm.shape}')
print(f'tris.shape == {tris.shape}')
print(f'rr_mm.max() = {rr_mm.max()}')  # just to show that we are in mm