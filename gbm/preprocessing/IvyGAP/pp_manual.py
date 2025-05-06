###############################################################################
# 
# This script post-processes IvyGAP data that has been manually registered.
# In particular:
#   The first visit T1 image is registered to the FreeSurfer T1 image.
#   Enhancing and Non-enhancing ROIs are combined to generate a single ROI.
#   ADC and ROI images are combined to generate a cellularity map.
#   The ADC, ROI, and cellularity map are transformed to the FreeSurfer space.
#  
# NOTE
#   Registered files will be saved in-place.
#   The updated patient information will be saved in the patient directory as "patient_info.json".
# 
# Assumptions:
#   All images have already been registered to a common space (likely the first visit pre-contrast T1 space).
#   Seperate ROIs exist for enhancing and non-enhancing tumor regions.
#   All images are in NIfTI format.
#   FreeSurfer's `recon-all` command has been run on the T1 image to generate the geometry.
#   The $SUBJECTS_DIR environment variable is set to the FreeSurfer subjects directory.
# 
# Usage: python3 pp_manual.py --pdir /path/to/patient/data
#   --pinfo /path/to/patient/info/json/file
#   --subjid FreeSurfer_subject_id
#   --elxparams $GEMINI_PATH/gbm/preprocessing/elastixParameters
# 
# todo: compute deformation fields
# 
###############################################################################

import os
import sys
import argparse
from pathlib import Path
import json

import itk
import nibabel

def main(args) -> None:
    
    # ---------------------------------------------------------------
    # General setup.
    # ---------------------------------------------------------------
    
    NUM_THREADS = args.nthreads
    VERBOSE = args.verbose
    PATIENT_DIR = args.pdir
    OUTPUT_FNAME = "patient_info.json"
    
    print(f"Patient directory: {args.pdir}")
    print(f"Patient information file: {args.pinfo}")
    
    with open(args.pinfo) as f:
        pinfo = json.load(f)
    
    # ---------------------------------------------------------------
    # Build registration from first visit to FreeSurfer space.
    # ---------------------------------------------------------------
    print("Registering reference T1 image to FreeSurfer space.")
    print(f"Elastix parameter directory: {args.elxparams}")
    RIGID_PARAMS = itk.ParameterObject.New()
    RIGID_PARAMS.ReadParameterFile(os.path.join(args.elxparams, "Parameters_Rigid.txt"))
    
    # load the images.
    fs_t1_path = os.path.join(os.getenv("SUBJECTS_DIR"), args.subjid, "nii", "orig_ras.nii")
    print(f"FreeSurfer T1 image: {fs_t1_path}")
    fs_t1 = itk.imread(fs_t1_path, itk.F)  # fixed image
    
    ref_t1_path = os.path.join(PATIENT_DIR, Path(pinfo['T1_pre']).name)
    print(f"Reference T1 image: {ref_t1_path}")
    ref_t1 = itk.imread(ref_t1_path, itk.F)  # moving image

    # build the registration object.
    ref_to_fs_img, ref_to_fs_rtp = itk.elastix_registration_method(fs_t1, ref_t1, parameter_object=RIGID_PARAMS, log_to_console=VERBOSE, number_of_threads=NUM_THREADS)
    ref_to_fs_rtp.WriteParameterFile(ref_to_fs_rtp, os.path.join(PATIENT_DIR, "ref_to_fs_transform_parameters.txt"))
    itk.imwrite(ref_to_fs_img, os.path.join(PATIENT_DIR, "ref_to_fs.nii"))
    
    # ---------------------------------------------------------------
    # Transform images to FreeSurfer space.
    # ---------------------------------------------------------------
    print("Transforming ADC and ROIs to FreeSurfer space.")
    for i, visit in enumerate(pinfo['visits']):
        # transform the ADC.
        adc = itk.imread(os.path.join(PATIENT_DIR, Path(visit['adc']).name), itk.F)
        adc_to_fs_img = itk.transformix_filter(adc, ref_to_fs_rtp)
        itk.imwrite(adc_to_fs_img, os.path.join(PATIENT_DIR, f"{Path(visit['adc']).stem}_fs.nii"))
        
        # transform the enhancing ROI.
        ref_to_fs_rtp.SetParameter(0, "ResampleInterpolator", "FinalNearestNeighborInterpolator")  # preserve label values
        roi_e = itk.imread(os.path.join(PATIENT_DIR, Path(visit['roi_enhance']).name), itk.F)
        roi_e_to_fs_img = itk.transformix_filter(roi_e, ref_to_fs_rtp)
        itk.imwrite(roi_e_to_fs_img, os.path.join(PATIENT_DIR, f"{Path(visit['roi_enhance']).stem}_fs.nii"))
        
        # transform the non-enhancing ROI.
        roi_ne = itk.imread(os.path.join(PATIENT_DIR, Path(visit['roi_nonenhance']).name), itk.F)
        roi_ne_to_fs_img = itk.transformix_filter(roi_ne, ref_to_fs_rtp)
        itk.imwrite(roi_ne_to_fs_img, os.path.join(PATIENT_DIR, f"{Path(visit['roi_nonenhance']).stem}_fs.nii"))
        
        ref_to_fs_rtp.SetParameter(0, "ResampleInterpolator", "FinalBSplineInterpolator")  # restore for next T1
    
    # ---------------------------------------------------------------
    # Combine ROIs.
    # ---------------------------------------------------------------
    print("Combining enhancing and non-enhancing ROIs.")
    for i, visit in enumerate(pinfo['visits']):
        j = i + 1  # 1-indexed visit numbers
        
        # load in the ROIs in FreeSurfer coordinates.
        roi_e = nibabel.load(os.path.join(PATIENT_DIR, f"{Path(visit['roi_enhance']).stem}_fs.nii"))
        roi_ne = nibabel.load(os.path.join(PATIENT_DIR, f"{Path(visit['roi_nonenhance']).stem}_fs.nii"))
        
        # combine the data.
        comb_data = roi_e.get_fdata() + roi_ne.get_fdata()
        outnii = nibabel.Nifti1Image(comb_data, affine=roi_e.affine, header=roi_e.header)
        
        ROI_NAME = f"ROI_v{j}_fs.nii"
        
        # write out the image.
        nibabel.save(outnii, os.path.join(PATIENT_DIR, ROI_NAME))
        
        # update the patient information.
        pinfo['visits'][i]['roi_fs'] = f"{os.path.join(Path(visit['adc']).parent, ROI_NAME)}"
    
    # ---------------------------------------------------------------
    # Compute tumor cellularity maps.
    # ---------------------------------------------------------------
    sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
    from dt4co.utils.data_utils import computeTumorCellularity, applyGaussianBlur
    
    print("Computing tumor cellularity maps.")
    for i, visit in enumerate(pinfo['visits']):
        j = i + 1  # 1-indexed visit numbers
        
        adc = nibabel.load(os.path.join(PATIENT_DIR, f"{Path(visit['adc']).stem}_fs.nii"))
        roi = nibabel.load(os.path.join(PATIENT_DIR, f"ROI_v{j}_fs.nii"))
        
        TUMOR_NAME = f"tumor_v{j}_fs.nii"
        TUMOR_BLUR_NAME = f"tumor_blur_v{j}_fs.nii"
        
        tumor = computeTumorCellularity(adc, roi)
        nibabel.save(tumor, os.path.join(PATIENT_DIR, TUMOR_NAME))
        
        # save a filtered version.
        tumor_blur = applyGaussianBlur(tumor)
        nibabel.save(tumor_blur, os.path.join(PATIENT_DIR, TUMOR_BLUR_NAME))
        
        # update the patient information.
        pinfo['visits'][i]['tumor_fs'] = f"{os.path.join(Path(visit['adc']).parent, TUMOR_NAME)}"
        
        # update the patient information.
        pinfo['visits'][i]['tumor_blur_fs'] = f"{os.path.join(Path(visit['adc']).parent, TUMOR_BLUR_NAME)}"
        
    
    # write updated patient information to file.
    with open(os.path.join(PATIENT_DIR, OUTPUT_FNAME), "w") as f:
        json.dump(pinfo, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process IvyGAP data that has been manually registered.")
    
    parser.add_argument("--pdir", type=str, help="Path to the patient directory.")
    parser.add_argument("--pinfo", type=str, help="Path to the patient information JSON file.")
    parser.add_argument("--elxparams", type=str, help="Path to the Elastix parameter files directory.")
    parser.add_argument("--subjid", type=str, help="Subject ID for FreeSurfer.")
    
    parser.add_argument("--nthreads", type=int, default=8, help="Number of threads to use.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Print verbose output.")
    
    parser.add_argument("--subdir", type=str, default=None, help="Subjects directory for FreeSurfer (for use when unable to set from environment variables).")
    
    args = parser.parse_args()
    
    main(args)
