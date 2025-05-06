#!/bin/bash

# ------------------------------------------------------------------------------
# Set paths.
# ------------------------------------------------------------------------------
PHANTOM_FS_DIR=/storage/graham/data/mri2fem-dataset/freesurfer/ernie/
PATIENT_FS_DIR=/storage/graham/data/subjects/nii_W36_1999_03_12/
PATIENT_DATA_DIR=/storage/graham/data/IvyGAP/W36/
PARAM_DIR=./elastixParameters/

# Get the imaging dates.
IMGDATES=($PATIENT_DATA_DIR/*)
BASELINE_DIR=${IMGDATES[0]}

# ------------------------------------------------------------------------------
# Get a NIftI image of the phantom.
# ------------------------------------------------------------------------------
if [ ! -f "$PHANTOM_FS_DIR/nii/orig_ras.nii" ]; then
    echo "'orig_ras.nii' NIfTI file not found. Converting..."
    mkdir -p "$PHANTOM_FS_DIR/nii/"
    
    # Convert the FreeSurfer image to NIfTI format with standard RAS-orientation.
    mri_convert "$PHANTOM_FS_DIR/mri/orig.mgz" "$PHANTOM_FS_DIR/nii/orig_ras.nii" --out_orientation RAS
fi

# ------------------------------------------------------------------------------
# Register the patient to the phantom.
# ------------------------------------------------------------------------------
echo "Building registration map between patient and phantom..."
mkdir -p "$BASELINE_DIR/reg/phantom/"
elastix -f "$PHANTOM_FS_DIR/nii/orig_ras.nii" -m "$PATIENT_FS_DIR/nii/orig_ras.nii" -p "$PARAM_DIR/Parameters_Rigid.txt" -out "$BASELINE_DIR/reg/phantom/"

# ------------------------------------------------------------------------------
# Apply the transform to the patient's data.
# ------------------------------------------------------------------------------
echo "Applying transform to patient data to bring it to phantom space..."
for date in ${IMGDATES[@]}; do
    REG_DIR="$BASELINE_DIR/reg/$(basename ${date})"
    echo "$REG_DIR"
    mkdir -p "$REG_DIR/phantom/"

    # Apply the transform to the cell denisty map to bring it to phantom space.
    transformix -in "$REG_DIR/celldensity_FS.nii" -def all -out "$REG_DIR/phantom/" -tp "$BASELINE_DIR/reg/phantom/TransformParameters.0.txt"
    
    # Rename the output file.
    mv $REG_DIR/phantom/result.nii $REG_DIR/phantom/celldensity_phantom.nii
    cp $REG_DIR/phantom/celldensity_phantom.nii $REG_DIR/celldensity_phantom.nii
done

# Shift the origin of the FreeSurface surfaces to align with the native T1 space.
echo "Shifting the surfaces..."
python3 applyRASShiftAllSurfaces.py --surfdir "$PHANTOM_FS_DIR/surf/" --t1 "$PHANTOM_FS_DIR/nii/orig_ras.nii" --no-ventricles

# Convert the gray/white matter segmentations to STL format.
echo "Converting surfaces to STL format..."
mris_convert "$PHANTOM_FS_DIR/surf/rh.pial.shifted" "$PHANTOM_FS_DIR/stl/rh.pial.shifted.stl"
mris_convert "$PHANTOM_FS_DIR/surf/lh.pial.shifted" "$PHANTOM_FS_DIR/stl/lh.pial.shifted.stl"
mris_convert "$PHANTOM_FS_DIR/surf/rh.white.shifted" "$PHANTOM_FS_DIR/stl/rh.white.shifted.stl"
mris_convert "$PHANTOM_FS_DIR/surf/lh.white.shifted" "$PHANTOM_FS_DIR/stl/lh.white.shifted.stl"
