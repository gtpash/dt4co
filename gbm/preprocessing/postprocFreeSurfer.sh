#!/bin/bash

##################################################################################################
# 
# This script post-processes the output from FreeSurfer's recon-all pipeline.
# Specifically, this script will:
#  1. Convert the FreeSurfer image to NIfTI format with standard RAS-orientation.
#  2. Extract the ventricles from the FreeSurfer segmentation.
#  3. Optionally, post-process the ventricles to remove small clusters and smooth the surface.
#  4. Shift the origin of the FreeSurfer surfaces to align with the native T1 space.
# 
# Usage: ./postprocFreeSurfer.sh <subject ID> <T1 image> <optional, subjects directory>
# 
##################################################################################################

if [[ $# -eq 0 ]] ; then
    echo "ERROR: No subject ID specified."
    exit 1
fi

# Subject ID to process.
SUBJID=${1}
T1=${2}
SD=${3:-$SUBJECTS_DIR}  # to override the default SUBJECTS_DIR environment variable

# Make directories to store results.
WORKING_DIR="$SD/$SUBJID"
mkdir -p "$WORKING_DIR/stl/"
mkdir -p "$WORKING_DIR/nii/"

# Convert the FreeSurfer image to NIfTI format with standard RAS-orientation.
echo "Converting FreeSurfer image to NIfTI format..."
echo "Saved to: $WORKING_DIR/nii/orig_ras.nii"
mri_convert "$WORKING_DIR/mri/orig.mgz" "$WORKING_DIR/nii/orig_ras.nii" --out_orientation RAS

# Input and output filenames for the ventricles.
INPUT="$WORKING_DIR/mri/wmparc.mgz"
OUTPUT="$WORKING_DIR/surf/ventricles"

# Also match the 4th ventricle and aqueduct?
INCLUDE_FOURTH_AND_AQUEDUCT=true

# Other parameters for ventricle processing.
POSTPROCESS=true
NUM_SMOOTHING=3
NUM_CLOSING=2
V_MIN=100

if [ "$INCLUDE_FOURTH_AND_AQUEDUCT" == true ]; then
    MATCHVAL="15"
else
    MATCHVAL="1"
fi

if [ "$POSTPROCESS" == true ]; then
    mri_binarize --i $INPUT --ventricles \
        --o "tmp.mgz"

    mri_volcluster --in "tmp.mgz" \
        --thmin 1 \
        --minsize $V_MIN \
        --ocn "tmp-ocn.mgz"

    mri_binarize --i "tmp-ocn.mgz" \
        --match 1 \
        --o "tmp.mgz"

    mri_morphology "tmp.mgz" \
        close $NUM_CLOSING "tmp.mgz"

    mri_binarize --i "tmp.mgz" \
        --match 1 \
        --surf-smooth $NUM_SMOOTHING \
        --surf $OUTPUT

    rm tmp.mgz
    rm tmp-ocn.mgz
    rm *.lut
else
    mri_binarize --i $INPUT --ventricles \
        --match $MATCHVAL \
        --surf-smooth $NUM_SMOOTHING \
        --surf $OUTPUT
fi

# Shift the origin of the FreeSurface surfaces to align with the native T1 space.
echo "Shifting surfaces to align with native T1 space..."
echo "Provided T1: $T1"
python3 $DT4CO_PATH/gbm/preprocessing/applyRASShiftAllSurfaces.py --surfdir "$WORKING_DIR/surf/" --t1 $T1

# Convert the gray/white matter segmentations to STL format.
echo "Converting surfaces to STL format..."
mris_convert "$WORKING_DIR/surf/rh.pial.shifted" "$WORKING_DIR/stl/rh.pial.shifted.stl"
mris_convert "$WORKING_DIR/surf/lh.pial.shifted" "$WORKING_DIR/stl/lh.pial.shifted.stl"
mris_convert "$WORKING_DIR/surf/rh.white.shifted" "$WORKING_DIR/stl/rh.white.shifted.stl"
mris_convert "$WORKING_DIR/surf/lh.white.shifted" "$WORKING_DIR/stl/lh.white.shifted.stl"
mris_convert "$WORKING_DIR/surf/ventricles.shifted" "$WORKING_DIR/stl/ventricles.shifted.stl"
