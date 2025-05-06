#!/bin/bash

##################################################################################################
# This script is meant to perfrom the entire pre-processing pipeline for IvyGAP data.
# 
# This script performs the following steps:
# 1. Convert DICOM files to NIfTI files using dcm2niix.
# 2. Rigidly register the baseline T1 image to longitudinal T1 images using elastix.
# 3. Compute inverse transform for rigid registration using elastix.
# 4. Deformably register the baseline T1 image to longitudinal T1 images using elastix, composing with the rigid transform.
# 5. Register the apparent diffusion coefficient (ADC) maps to the native T1 image space.
# 6. Call SimpleITK to transform the OncoHabitats segmentations to the native T1 image space.
# 7. Apply the inverse rigid transformation to the longitduinal segmentations to bring them into the baseline T1 image space.
# 8. Register the baseline T1 image to FreeSurfer space using elastix.
# 
# Assumptions:
# The chronological first imaging date is the baseline.
# The OncoHabitats segmentation registered to the native (day of imaging) T1 image space is named: Segmentation_T1.nii.
# You have an appropriate conda environment activated.
# 
# Usage: ./runElastixPipeline.sh
#
##################################################################################################

# ----------------------------------------
# 0. Set paths.
# ----------------------------------------
PATIENTDIR=/storage/graham/data/IvyGAP/W36/
PARAMETERDIR=../elastixParameters/
SUBJID="nii_W36_1999_03_12"
SUBDIR="$SUBJECTS_DIR/$SUBJID"

# Get the imaging dates.
IMGDATES=($PATIENTDIR/*)
BASELINE=${IMGDATES[0]}

echo "Imaging dates: ${IMGDATES[@]}"
echo "Baseline imaging directory: ${BASELINE}"
echo "Elastix parameter directory: ${PARAMETERDIR}"

counter=1  # hack to ensure you don't register the baseline to itself.

for date in ${IMGDATES[@]}; do
    # ----------------------------------------
    # 1. Call script to convert DICOM files to NIfTI files using dcm2niix.
    # ----------------------------------------
    ./convertDCM2NII.sh $date

    V2=$date

    # ----------------------------------------
    # Steps 2-4. Compute (inverse) rigid and deformable registrations.
    # Only apply longitudinal registration if this is not the baseline.
    # ----------------------------------------
    OUTDIR=$BASELINE/reg/$(basename ${V2})
    if [ "${counter}" -eq "1" ]; then
        : # no-op
    else
        # Grab the name of the T1 you're registering to.
        ../elastixLongitudinalRegistration.sh $BASELINE $V2 $OUTDIR $PARAMETERDIR
    fi

    # ----------------------------------------
    # 5. Register the ADC maps to the native T1 image space.
    # ----------------------------------------
    echo "Registering ADC maps to native T1 image space..."
    mkdir -p $OUTDIR/ADC_T1/
    elastix -f ${V2}/nii/T1.nii -m ${V2}/nii/ADC.nii -p $PARAMETERDIR/Parameters_Rigid.txt -out $OUTDIR/ADC_T1/
    mv $OUTDIR/ADC_T1/result.0.nii $OUTDIR/ADC_T1/ADC_T1.nii

    # ----------------------------------------
    # 6. Register the OncoHabitats segmentations to the native T1 image space.
    # ----------------------------------------
    echo "Registering OncoHabitats segmentations to native T1 image space..."
    python3 ../applySegmentationTransformOH.py --transform ${V2}/oncohabitats/transforms/transform_t1_to_t1c_0GenericAffine.mat --moving ${V2}/oncohabitats/native/Segmentation.nii --fixed ${V2}/nii/T1.nii --outdir ${V2}/nii/

    # ----------------------------------------
    # 7. Apply inverse transform to OncoHabitats segmentations to bring them to baseline T1 space.
    # Only apply longitudinal registration if this is not the baseline.
    # ----------------------------------------
    echo "Applying inverse transform to OncoHabitats segmentations to bring them to baseline T1 space..."
    if [ "${counter}" -eq "1" ]; then
        : # no-op
    else
        mkdir -p $OUTDIR/tumorseg/
        transformix -in ${V2}/nii/Segmentation_T1.nii -def all -out $OUTDIR/tumorseg -tp $OUTDIR/rigidInverse/TransformParameters.0.txt
        mv $OUTDIR/tumorseg/result.nii $OUTDIR/tumorseg/Segmentation_T1.nii
    fi

    # ----------------------------------------
    # 8. Register the baseline T1, segmentations, ADC to FreeSurfer space.
    # ----------------------------------------
    mkdir -p $OUTDIR/tumorseg_FS/
    
    # Apply the transform to the segmentations to bring them to FreeSurfer space.
    if [ "${counter}" -eq "1" ]; then
        echo "Registering baseline T1 to FreeSurfer space..."
        
        mkdir -p $BASELINE/reg/freesurfer/
        mkdir -p $BASELINE/reg/freesurferInverse/
        mkdir -p $SUBDIR/nii/

        # Convert the FreeSurfer image to NIfTI format with standard RAS-orientation.
        mri_convert "$SUBDIR/mri/orig.mgz" "$SUBDIR/nii/orig_ras.nii" --out_orientation RAS

        # Register the baseline T1 to FreeSurfer space.
        elastix -f "$SUBDIR/nii/orig_ras.nii" -m "$BASELINE/nii/T1.nii" -p "$PARAMETERDIR/Parameters_Affine.txt" -out "$BASELINE/reg/freesurfer/"
        
        # Make a copy of the transform and set ResampleInterpolator to "FinalNearestNeighborInterpolator".
        cp "$BASELINE/reg/freesurfer/TransformParameters.0.txt" "$BASELINE/reg/freesurfer/TransformParameters.NNInterpolator.txt"
        sed -i 's/^(ResampleInterpolator.*/(ResampleInterpolator "FinalNearestNeighborInterpolator")/' "$BASELINE/reg/freesurfer/TransformParameters.NNInterpolator.txt"

        # Compute the inverse of the transform.
        echo "Computing inverse of the FreeSurfer transform..."
        elastix -f "$SUBDIR/nii/orig_ras.nii" -m "$SUBDIR/nii/orig_ras.nii" -p $PARAMETERDIR/Parameters_InvAffine.txt -out $BASELINE/reg/freesurferInverse/ -t0 $BASELINE/reg/freesurfer/TransformParameters.0.txt
        # Set Initial Transform to "NoInitialTransform" in the inverse.
        sed -i 's/^(InitialTransform.*/(InitialTransform "NoInitialTransform")/' $OUTDIR/freesurferInverse/TransformParameters.0.txt

        # Apply the transform to the segmentation to bring it to FreeSurfer space.
        transformix -in $BASELINE/nii/Segmentation_T1.nii -def all -out $OUTDIR/tumorseg_FS/ -tp $BASELINE/reg/freesurfer/TransformParameters.NNInterpolator.txt
    else
        # Apply the transform to the segmentations to bring it to FreeSurfer space.
        transformix -in $OUTDIR/tumorseg/Segmentation_T1.nii -def all -out $OUTDIR/tumorseg_FS/ -tp $BASELINE/reg/freesurfer/TransformParameters.NNInterpolator.txt
    fi

    # Rename the output file.
    mv $OUTDIR/tumorseg_FS/result.nii $OUTDIR/tumorseg_FS/Segmentation_FS.nii

    # Apply the transform to the ADC map to bring it to FreeSurfer space.
    mkdir -p $OUTDIR/ADC_FS/
    transformix -in $OUTDIR/ADC_T1/ADC_T1.nii -def all -out $OUTDIR/ADC_FS/ -tp $BASELINE/reg/freesurfer/TransformParameters.0.txt
    mv $OUTDIR/ADC_FS/result.nii $OUTDIR/ADC_FS/ADC_FS.nii

    counter=$((counter +1))

done
