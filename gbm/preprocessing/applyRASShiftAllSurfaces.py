################################################################################
# 
# This script applies the C_(RAS) shift to all FreeSurfer derived surfaces.
# 
# Usage: python3 applyRASShiftAllSurfaces.py --surfdir /path/to/surfaces --t1 /path/to/T1
# 
################################################################################

import os
import argparse

import nibabel
import numpy as np

def applyFreeSurferShiftRAS(surfdir:str, surf:str, t1:str=None):
    """Apply FreeSurfer C_(RAS) shift to surface.

    Args:
        surfdir (str): Absolute path to the directory with FreeSurfer surface files.
        surf (str): Name of surface file.
        t1 (str, optional): Absolute path to T1 image. If provided, will use the center voxel to RAS shift from the T1 image. Defaults to None.
    """
    # Load in the surface.
    coords, faces, header = nibabel.freesurfer.io.read_geometry(os.path.join(surfdir, surf), read_metadata=True)
    
    if t1 is not None:
        t1 = nibabel.load(t1)
        mgh = nibabel.MGHImage(t1.dataobj, t1.affine)
        cras = nibabel.affines.apply_affine(mgh.header.get_vox2ras(), np.array(mgh.get_fdata().shape) // 2)
        coords += cras
    else:
        # Update coordinates, zero out the shift in the header.
        coords += header['cras']
    
    header['cras'] = np.array([0., 0., 0.])
    
    # Write out the shifted surface.
    nibabel.freesurfer.io.write_geometry(os.path.join(surfdir, f"{surf}.shifted"), coords, faces, volume_info=header)


def main(args):
    """
    Shift FreeSurfer derived surfaces by the C_(RAS) shift specified in the header of the FreeSurfer image.
    """
    
    if args.ventricles:
        SURFS = ["lh.pial", "rh.pial", "lh.white", "rh.white", "ventricles"]
    else:
        SURFS = ["lh.pial", "rh.pial", "lh.white", "rh.white"]
    
    for surf in SURFS:
        if args.t1 is not None:
            applyFreeSurferShiftRAS(args.surfdir, surf, args.t1)
        else:
            applyFreeSurferShiftRAS(args.surfdir, surf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shift FreeSurfer derived surfaces by the C_(RAS) shift specified in the header of the FreeSurfer image.")

    parser.add_argument("--surfdir", type=str, help="Absolute path to the directory with FreeSurfer surface files.")
    parser.add_argument("--t1", type=str, default=None, help="Absolute path to the T1 image in native space.")
    parser.add_argument("--ventricles", action=argparse.BooleanOptionalAction, default=True, help="Should the ventricles be processed?")
    
    args = parser.parse_args()
    main(args)
