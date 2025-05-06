################################################################################
# This script runs the longitudinal registration pipeline for the IvyGAP dataset.
#
# Assumptions:
# The chronological first scan is the reference / baseline scan.
# The typical FreeSurfer paths should be set, in particular the SUBJECTS_DIR environment variable.
# Raw data has already been converted to NIfTI format and is available in the </path/to/imaging/date>/nii directory.
#   If the raw data has not already been converted to NIfTI format, please use
#   the `dcm2niix` tool to convert the data or the "convertDCM2NII.sh" script.
# 
# Usage: python3 runLongitudinalRegistration.py --pdir /path/to/patient/data --subjid FreeSurferSubjectID --elxparams /path/to/elastix/parameters
# 
# NOTE: This script could be modified to use default itk-elastix parameters
#      however, we load from a directory to allow for easy customization and
#      compatibility with the compiled Elastix binaries.
# 
################################################################################

import os
import argparse
import itk
import SimpleITK as sitk

def load_param_files(param_dir: str) -> tuple:
    RIGID_PARAMS = itk.ParameterObject.New()
    RIGID_PARAMS.ReadParameterFile(os.path.join(param_dir, "Parameters_Rigid.txt"))
    
    INV_RIGID_PARAMS = itk.ParameterObject.New()
    INV_RIGID_PARAMS.ReadParameterFile(os.path.join(param_dir, "Parameters_InvRigid.txt"))
    
    AFFINE_PARAMS = itk.ParameterObject.New()
    AFFINE_PARAMS.ReadParameterFile(os.path.join(param_dir, "Parameters_Affine.txt"))
    
    INV_AFFINE_PARAMS = itk.ParameterObject.New()
    INV_AFFINE_PARAMS.ReadParameterFile(os.path.join(param_dir, "Parameters_InvAffine.txt"))
    
    BSPLINE_PARAMS = itk.ParameterObject.New()
    BSPLINE_PARAMS.ReadParameterFile(os.path.join(param_dir, "Parameters_BSpline.txt"))
    
    return RIGID_PARAMS, INV_RIGID_PARAMS, AFFINE_PARAMS, INV_AFFINE_PARAMS, BSPLINE_PARAMS


def main(args) -> None:
    # ---------------------------------------------------------------
    # Unpack arguments and build necessary paths.
    # ---------------------------------------------------------------
    VERBOSE = args.verbose
    PATIENT_DIR = args.pdir
    SUBJECT_FS_DIR = os.path.join(os.getenv("SUBJECTS_DIR"), args.subjid)
    ELASTIX_PARAM_DIR = args.elxparams
    RIGID_PARAMS, INV_RIGID_PARAMS, AFFINE_PARAMS, INV_AFFINE_PARAMS, BSPLINE_PARAMS = load_param_files(ELASTIX_PARAM_DIR)
    NUM_THREADS = args.nthreads
    
    dates = [f.name for f in os.scandir(PATIENT_DIR) if f.is_dir()]
    dates.sort()  # just to be sure
    
    print(f"Patient directory: {PATIENT_DIR}")
    print(f"FreeSurfer subject directory: {SUBJECT_FS_DIR}")
    print(f"Found {len(dates)} imaging dates.")
    print(f"Imaging dates: {dates}")
    
    for i, date in enumerate(dates):
        CURRENT_DATE_DIR = os.path.join(PATIENT_DIR, date)
        
        if i == 0:
            REFERENCE_DIR = CURRENT_DATE_DIR
            OUTPUT_DIR = os.path.join(REFERENCE_DIR, "reg", date)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # This is the reference scan.
            # We will use this as the baseline for the longitudinal registration.
            FS_REG_DIR = os.path.join(REFERENCE_DIR, "reg", "fs")
            os.makedirs(FS_REG_DIR, exist_ok=True)
            
            # ---------------------------------------------------------------
            # Build registration to the FreeSurfer T1 image space.
            # ---------------------------------------------------------------
            ref_t1_path = os.path.join(REFERENCE_DIR, "nii", "T1.nii")
            ref_t1 = itk.imread(ref_t1_path, itk.F)        # moving image
            fs_t1 = itk.imread(os.path.join(SUBJECT_FS_DIR, "nii", "orig_ras.nii"), itk.F)  # fixed image
            ref_to_fs_img, ref_to_fs_rtp = itk.elastix_registration_method(fs_t1, ref_t1, parameter_object=RIGID_PARAMS, log_to_console=VERBOSE, number_of_threads=NUM_THREADS)
            ref_to_fs_rtp.WriteParameterFile(ref_to_fs_rtp, os.path.join(FS_REG_DIR, "ref_to_fs_transform_parameters.txt"))
            
        else:
            # ---------------------------------------------------------------
            # Longitudinal registration.
            # 
            # Specifically, we will:
            # a. Rigidly register the reference scan (moving) to the current scan (fixed).
            # b. Compute the inverse of the transformation, by rigidly registering the current scan (moving) to the reference scan (fixed).
            # c. Deformably register the reference scan to the rigidly registered current scan.
            #   c0) in the reference image coordinates.
            #   c1) in the current image coordinates.
            # d. Compute the associated deformation field (with transformix)
            #   in the reference image coordinates.
            # 
            # NOTE: Elastix also writes the transform parameters to .txt file if an output directory is specified.
            # todo: some parts of the registration are not necessary, but are included for completeness.
            # ---------------------------------------------------------------
            
            OUTPUT_DIR = os.path.join(REFERENCE_DIR, "reg", date)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            cur_t1_path = os.path.join(CURRENT_DATE_DIR, "nii", "T1.nii")
            cur_t1 = itk.imread(cur_t1_path, itk.F)     # fixed image
            # ref_t1 should have already been loaded.
            
            print("Rigid registration of reference scan to current scan.")
            print(f"Reference scan: {ref_t1_path}")
            print(f"Current scan: {cur_t1_path}")
            
            # Rigidly register the reference scan to the current scan, save the result and transform.
            ref_to_cur_img, ref_to_cur_rtp = itk.elastix_registration_method(cur_t1, ref_t1, parameter_object=RIGID_PARAMS, log_to_console=VERBOSE, number_of_threads=NUM_THREADS)
            itk.imwrite(ref_to_cur_img, os.path.join(OUTPUT_DIR, "rigid_ref_to_cur_result.nii"))
            ref_to_cur_rtp.WriteParameterFile(ref_to_cur_rtp, os.path.join(OUTPUT_DIR, "rigid_ref_to_cur_transform_parameters.txt"))
            
            # Rigidly register the current scan to the reference scan, save the result and transform.
            cur_to_ref_img, cur_to_ref_rtp = itk.elastix_registration_method(ref_t1, cur_t1, parameter_object=RIGID_PARAMS, log_to_console=VERBOSE, number_of_threads=NUM_THREADS)
            itk.imwrite(cur_to_ref_img, os.path.join(OUTPUT_DIR, "rigid_cur_to_ref_result.nii"))
            cur_to_ref_rtp.WriteParameterFile(cur_to_ref_rtp, os.path.join(OUTPUT_DIR, "rigid_cur_to_ref_transform_parameters.txt"))
            
            # Deformable registration of the reference scan (in current scan coordinates) to the current scan.
            # todo: add a ROI + margin mask to the deformable registration.
            _, deform_rtp_cur_coords = itk.elastix_registration_method(cur_t1, ref_to_cur_img, parameter_object=BSPLINE_PARAMS, log_to_console=VERBOSE, number_of_threads=NUM_THREADS)
            deform_rtp_cur_coords.WriteParameterFile(deform_rtp_cur_coords, os.path.join(OUTPUT_DIR, "bspline_cur_coords_transform_parameters.txt"))
            
            # todo: project deformation field onto mesh and check validity.
            # compute the deformable registration of the reference scan in the FreeSurfer coordinates.
            cur_to_fs_img = itk.transformix_filter(cur_to_ref_img, ref_to_fs_rtp)
            _, deform_rtp_ref_coords = itk.elastix_registration_method(cur_to_fs_img, ref_to_fs_img, parameter_object=BSPLINE_PARAMS, log_to_console=VERBOSE, number_of_threads=NUM_THREADS)
            deform_rtp_ref_coords.WriteParameterFile(deform_rtp_ref_coords, os.path.join(OUTPUT_DIR, "bspline_fs_coords_transform_parameters.txt"))
            
            # compute the deformation field of the reference scan in the reference scan coordinates.
            def_field_bspline_ref_coords = itk.transformix_deformation_field(ref_t1, deform_rtp_ref_coords)
            itk.imwrite(def_field_bspline_ref_coords, os.path.join(OUTPUT_DIR, "deformationField_bspline_fs_coords.nii"))
    
        # ---------------------------------------------------------------
        # Register the ADC scan to the T1 image space and then transform to FreeSurfer space.
        # ---------------------------------------------------------------
        cur_t1 = itk.imread(os.path.join(CURRENT_DATE_DIR, "nii", "T1.nii"), itk.F)     # fixed image
        cur_adc = itk.imread(os.path.join(CURRENT_DATE_DIR, "nii", "ADC.nii"), itk.F)   # moving image
        adc_to_t1_img, adc_to_t1_rtp = itk.elastix_registration_method(cur_t1, cur_adc, parameter_object=RIGID_PARAMS, log_to_console=VERBOSE, number_of_threads=NUM_THREADS)
        itk.imwrite(adc_to_t1_img, os.path.join(OUTPUT_DIR, "ADC_ref.nii"))
        adc_to_t1_rtp.WriteParameterFile(adc_to_t1_rtp, os.path.join(OUTPUT_DIR, "ADC_T1_transform_parameters.txt"))
        
        # transform ADC scan to Fresurfer space.
        if i == 0:
            # only one hop to FreeSurfer space.
            adc_fs = itk.transformix_filter(adc_to_t1_img, ref_to_fs_rtp)
        else:
            # current T1 space -> reference T1 space
            adc_ref = itk.transformix_filter(adc_to_t1_img, cur_to_ref_rtp)
            # reference T1 space -> FreeSurfer space
            adc_fs = itk.transformix_filter(adc_ref, ref_to_fs_rtp)
        
        itk.imwrite(adc_fs, os.path.join(OUTPUT_DIR, "ADC_FS.nii"))

        # ----------------------------------------
        # Register the OncoHabitats segmentations to the T1 image space and then transform to FreeSurfer space.
        # 
        # NOTE: OncoHabitats uses ANTsX for registration, so we need to use the ANTsX transform to move the segmentations to the T1 image space.
        # ----------------------------------------
        # Load the transform file and images.
        tf = sitk.ReadTransform(os.path.join(CURRENT_DATE_DIR, "oncohabitats", "transforms", "transform_t1_to_t1c_0GenericAffine.mat"))
        oh_fix = sitk.ReadImage(os.path.join(CURRENT_DATE_DIR, "nii", "T1.nii"))
        oh_mov = sitk.ReadImage(os.path.join(CURRENT_DATE_DIR, "oncohabitats", "native", "Segmentation.nii"))
        
        # Resample with the inverse transform.
        reg = sitk.Resample(oh_mov, oh_fix, tf.GetInverse(), sitk.sitkNearestNeighbor, 0.0, oh_mov.GetPixelIDValue())
        
        # Write out the resampled (registered) image.
        sitk.WriteImage(reg, os.path.join(OUTPUT_DIR, "ROI_T1.nii"))
        
        # transform OncoHabitats segmentations to FreeSurfer space.
        oh_cur = itk.imread(os.path.join(OUTPUT_DIR, "ROI_T1.nii"), itk.F)  # moving image
        if i == 0:
            # only one hop to FreeSurfer space.
            ref_to_fs_rtp.SetParameter(0, "ResampleInterpolator", "FinalNearestNeighborInterpolator")  # preserve label values
            oh_fs = itk.transformix_filter(oh_cur, ref_to_fs_rtp)
            ref_to_fs_rtp.SetParameter(0, "ResampleInterpolator", "FinalBSplineInterpolator")
            itk.imwrite(oh_fs, os.path.join(OUTPUT_DIR, "ROI_FS.nii"))
        else:
            # current T1 space -> reference T1 space
            cur_to_ref_rtp.SetParameter(0, "ResampleInterpolator", "FinalNearestNeighborInterpolator")  # preserve label values
            oh_ref = itk.transformix_filter(oh_cur, cur_to_ref_rtp)
            cur_to_ref_rtp.SetParameter(0, "ResampleInterpolator", "FinalBSplineInterpolator")
            # reference T1 space -> FreeSurfer space
            ref_to_fs_rtp.SetParameter(0, "ResampleInterpolator", "FinalNearestNeighborInterpolator")  # preserve label values
            oh_fs = itk.transformix_filter(oh_ref, ref_to_fs_rtp)
            ref_to_fs_rtp.SetParameter(0, "ResampleInterpolator", "FinalBSplineInterpolator")
            itk.imwrite(oh_fs, os.path.join(OUTPUT_DIR, "ROI_FS.nii"))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run longitudinal registration")
    
    parser.add_argument("--pdir", type=str, help="Path to the patient data directory.")
    parser.add_argument("--subjid", type=str, help="FreeSurfer Subject ID.")
    parser.add_argument("--elxparams", type=str, help="Path to directory containing Elastix parameters.")
    
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose output.")
    parser.add_argument("--nthreads", type=int, default=8, help="Number of threads to use in Elastix calls.")
    
    args = parser.parse_args()
    main(args)
    