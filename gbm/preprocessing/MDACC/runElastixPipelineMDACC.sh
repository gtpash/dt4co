#!/bin/bash
# Need to convert DCM to NIfTI using dcm2niix
# Need to re-orient all of the images so that they are RAS

# follow "standard" pipeline?
# need to account for "enhancing" ROI_E and "non-enhancing" ROI_NE

# This script performs the following steps:
# 1. Convert DICOM files to NIfTI files using dcm2niix.
# 2. Rigidly register the baseline T1 image to longitudinal T1 images using elastix.
# 3. Compute inverse transform for rigid registration using elastix.
# 4. Deformably register the baseline T1 image to longitudinal T1 images using elastix, composing with the rigid transform.
# 5. Register the apparent diffusion coefficient (ADC) maps to the native T1 image space.
# 6. Call SimpleITK to transform the OncoHabitats segmentations to the native T1 image space.
# 7. Apply the inverse rigid transformation to the longitduinal segmentations to bring them into the baseline T1 image space.
# 8. Register the baseline T1 image to FreeSurfer space using elastix.

# Assumptions:
# The chronological first imaging date is the baseline.
# The OncoHabitats segmentation registered to the native (day of imaging) T1 image space is named: Segmentation_T1.nii.
# You have an appropriate conda environment activated.

# ----------------------------------------
# 0. Set paths.
# ----------------------------------------
PATIENTDIR=/storage/graham/data/MDA_CNS/2007_06_20/
PARAMETERDIR=../elastixParameters/
SUBJID="mda"
SUBDIR="$SUBJECTS_DIR/$SUBJID"

# Get the imaging dates.
IMGDATES=($PATIENTDIR/*)
BASELINE=${IMGDATES[0]}

echo "Imaging dates: ${IMGDATES[@]}"
echo "Baseline imaging directory: ${BASELINE}"
echo "Elastix parameter directory: ${PARAMETERDIR}"

counter=1  # hack to ensure you don't register the baseline to itself.