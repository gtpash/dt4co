################################################################################
# This script post-processes the output of the longitudinal registration pipeline.
# In particular:
#   - ADC and ROI images are combined to generate a cellularity map.
# 
# Usage: python3 pp_registration.py --pdir /path/to/patient/data
# 
# NOTE
#   This script is needed because `itk-elastix` conflicts with a dependency of `dt4co`.
# 
################################################################################

import os
import sys
import argparse

import nibabel

sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
from dt4co.utils.data_utils import computeTumorCellularity, applyGaussianBlur

def main(args)->None:
    
    # ---------------------------------------------------------------
    # Unpack arguments and build necessary paths.
    # ---------------------------------------------------------------
    
    PATIENT_DIR = args.pdir
    
    dates = [f.name for f in os.scandir(PATIENT_DIR) if f.is_dir()]
    dates.sort()  # just to be sure
    
    print(f"Patient directory: {PATIENT_DIR}")
    print(f"Found {len(dates)} imaging dates.")
    print(f"Imaging dates: {dates}")
    
    # get the image dates.
    dates = [f.name for f in os.scandir(PATIENT_DIR) if f.is_dir()]
    dates.sort()  # just to be sure

    # ---------------------------------------------------------------
    # Loop through the dates, compute the cellularity maps, and write them to file.
    # ---------------------------------------------------------------
    for i, date in enumerate(dates):
        
        # get the registration directory.
        if i == 0:
            REFERENCE_DIR = os.path.join(PATIENT_DIR, date)
        
        adc_img = nibabel.load(os.path.join(REFERENCE_DIR, "reg", date, "ADC_FS.nii"))
        roi_img = nibabel.load(os.path.join(REFERENCE_DIR, "reg", date, "ROI_FS.nii"))
        tumor = computeTumorCellularity(adc_img, roi_img)
        nibabel.save(tumor, os.path.join(REFERENCE_DIR, "reg", date, "tumor_FS.nii"))
        
        # save a filtered version.
        tumor_blur = applyGaussianBlur(tumor)
        nibabel.save(tumor_blur, os.path.join(REFERENCE_DIR, "reg", date, "tumor_FS_blur.nii"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the tumor cellularity maps and write them to file.")
    parser.add_argument("--pdir", type=str, help="Path to the patient data directory.")
    args = parser.parse_args()
    main(args)
