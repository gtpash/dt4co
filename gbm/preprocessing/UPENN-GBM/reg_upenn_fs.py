###############################################################################
# 
# This script post-processes UPENN-GBM data.
# In particular:
#   The T1 image is registered to the FreeSurfer T1 image.
#   ROI is transformed to the FreeSurfer space.
#  
# NOTE
#   Registered files will be saved in-place.
#   The updated patient information will be saved in the patient directory as "patient_info.json".
# 
# Assumptions:
#   All images have already been registered to a common space (likely the first visit pre-contrast T1 space).
#   All ROIs are combined in a single file.
#   All images are in NIfTI format.
#   FreeSurfer's `recon-all` command has been run on the T1 image to generate the geometry.
#   The $SUBJECTS_DIR environment variable is set to the FreeSurfer subjects directory.
# 
# Usage: python3 reg_upenn_fs.py --pdir /path/to/patient/data
#   --subid FreeSurfer_subject_id
#   --elxparams $DT4CO_PATH/gbm/preprocessing/elastixParameters
#   --subdir /path/to/freesurfer/subjects
# 
###############################################################################

import os
import sys
import argparse

import itk
import nibabel

def main(args) -> None:
    
    # ---------------------------------------------------------------
    # General setup.
    # ---------------------------------------------------------------
    
    NUM_THREADS = args.nthreads
    VERBOSE = args.verbose
    PATIENT_DIR = args.pdir
    SUBJECTS_DIR = args.subdir
    
    print(f"Patient directory: {PATIENT_DIR}")
    
    # names of the images
    SUBID = args.subid
    T1_FILE = f"UPENN-GBM-{SUBID}_11_T1_unstripped.nii"
    ROI_FILE = f"UPENN-GBM-{SUBID}_11_segm.nii"
    
    NONENHANCING_VAL = 0.16     # low cellularity for non-enhancing tumor
    ENHANCING_VAL = 0.8         # high cellularity for enhancing tumor
    
    # ---------------------------------------------------------------
    # Build registration from first visit to FreeSurfer space.
    # ---------------------------------------------------------------
    print("Registering reference T1 image to FreeSurfer space.")
    print(f"Elastix parameter directory: {args.elxparams}")
    RIGID_PARAMS = itk.ParameterObject.New()
    RIGID_PARAMS.ReadParameterFile(os.path.join(args.elxparams, "Parameters_Rigid.txt"))
    
    # load the images.
    fs_t1_path = os.path.join(SUBJECTS_DIR, f"sub-{SUBID}", "nii", "orig_ras.nii")
    print(f"FreeSurfer T1 image: {fs_t1_path}")
    fs_t1 = itk.imread(fs_t1_path, itk.F)  # fixed image
    
    ref_t1_path = os.path.join(PATIENT_DIR, T1_FILE)
    print(f"Reference T1 image: {ref_t1_path}")
    ref_t1 = itk.imread(ref_t1_path, itk.F)  # moving image

    # build the registration object.
    ref_to_fs_img, ref_to_fs_rtp = itk.elastix_registration_method(fs_t1, ref_t1, parameter_object=RIGID_PARAMS, log_to_console=VERBOSE, number_of_threads=NUM_THREADS)
    ref_to_fs_rtp.WriteParameterFile(ref_to_fs_rtp, os.path.join(PATIENT_DIR, "ref_to_fs_transform_parameters.txt"))
    itk.imwrite(ref_to_fs_img, os.path.join(PATIENT_DIR, "ref_to_fs.nii"))
    
    # ---------------------------------------------------------------
    # Transform images to FreeSurfer space.
    # ---------------------------------------------------------------
    print("Transforming the ROI to FreeSurfer space.")
    # transform the enhancing ROI.
    ref_to_fs_rtp.SetParameter(0, "ResampleInterpolator", "FinalNearestNeighborInterpolator")  # preserve label values
    roi = itk.imread(os.path.join(PATIENT_DIR, ROI_FILE), itk.F)
    roi_to_fs_img = itk.transformix_filter(roi, ref_to_fs_rtp)
    itk.imwrite(roi_to_fs_img, os.path.join(PATIENT_DIR, f"roi_fs.nii"))
    
    ref_to_fs_rtp.SetParameter(0, "ResampleInterpolator", "FinalBSplineInterpolator")  # restore for next T1
    
    # ---------------------------------------------------------------
    # Transform UPENN-GBM labels to enhancing / non-enhancing cellularities.
    # ---------------------------------------------------------------
    print("Writing out cellularity map.")
    roi_nii = nibabel.load(os.path.join(PATIENT_DIR, f"roi_fs.nii"))
    roi_data = roi_nii.get_fdata()
    
    # labels from the BraTS challenge (see: https://arxiv.org/abs/1811.02629)
    roi_data[roi_data == 1] = ENHANCING_VAL     # NCR
    roi_data[roi_data == 2] = NONENHANCING_VAL  # ED
    roi_data[roi_data == 3] = NONENHANCING_VAL  # NET
    roi_data[roi_data == 4] = ENHANCING_VAL     # AT
    
    # spoof the image
    outnii = nibabel.Nifti1Image(roi_data, affine=roi_nii.affine, header=roi_nii.header)
    
    # write the image
    nibabel.save(outnii, os.path.join(PATIENT_DIR, f"tumor_fs.nii"))
    
    # import after ITK to avoid HDF5 conflict
    sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
    from dt4co.utils.data_utils import applyGaussianBlur
    
    # apply a Gaussian filter to the cellularity.
    tumor_blur = applyGaussianBlur(outnii)
    nibabel.save(tumor_blur, os.path.join(PATIENT_DIR, f"tumor_blur_fs.nii"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process IvyGAP data that has been manually registered.")
    
    parser.add_argument("--pdir", type=str, help="Path to the patient directory.")
    parser.add_argument("--elxparams", type=str, help="Path to the Elastix parameter files directory.")
    parser.add_argument("--subid", type=str, help="Subject ID for FreeSurfer.")
    
    parser.add_argument("--nthreads", type=int, default=8, help="Number of threads to use.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Print verbose output.")
    
    parser.add_argument("--subdir", type=str, default=None, help="Subjects directory for FreeSurfer (for use when unable to set from environment variables).")
    
    args = parser.parse_args()
    
    main(args)
