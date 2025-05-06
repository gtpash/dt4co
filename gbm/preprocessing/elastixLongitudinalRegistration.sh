#!/bin/bash

# This script performs the following steps in the pre-processing pipleine, using Elastix:
# 2. Rigidly register the baseline T1 image to longitudinal T1 images using elastix.
# 3. Compute inverse transform for rigid registration using elastix.
# 4. Deformably register the baseline T1 image to longitudinal T1 images using elastix, composing with the rigid transform.

# Input arguments from the driver script.
BASELINE=${1}
V2=${2}
OUTDIR=${3} # this could be built on the fly, but we make the call explicit in the driver.
PARAMETERDIR=${4}

# ----------------------------------------
# 2. Rigidly register the baseline T1 image to longitudinal T1 images using elastix.
# ----------------------------------------

# Make directories for transforms.
mkdir -p $OUTDIR $OUTDIR/rigid/ $OUTDIR/deformable/ $OUTDIR/rigidInverse/

echo "Rigid registration of baseline T1 to visit 2 T1..."
echo "Results stored in ${OUTDIR}/rigid/"
elastix -f $V2/nii/T1.nii -m $BASELINE/nii/T1.nii -p $PARAMETERDIR/Parameters_Rigid.txt -out $OUTDIR/rigid/

# ----------------------------------------
# 3. Compute inverse of the rigid transform.
# ----------------------------------------
echo "Computing inverse rigid transform..."
echo "Results stored in ${OUTDIR}/rigidInverse/"
elastix -f $V2/nii/T1.nii -m $V2/nii/T1.nii -p $PARAMETERDIR/Parameters_InvRigid.txt -out $OUTDIR/rigidInverse/ -t0 $OUTDIR/rigid/TransformParameters.0.txt
# Set Initial Transform to "NoInitialTransform" in the inverse.
sed -i 's/^(InitialTransform.*/(InitialTransform "NoInitialTransform")/' $OUTDIR/rigidInverse/TransformParameters.0.txt
# Set ResampleInterpolator to "FinalNearestNeighborInterpolator" in the inverse.
sed -i 's/^(ResampleInterpolator.*/(ResampleInterpolator "FinalNearestNeighborInterpolator")/' $OUTDIR/rigidInverse/TransformParameters.0.txt

# ----------------------------------------
# 4. Deformable registration of baseline T1 image to longitudinal T1 images using elastix, composed  with the rigid transform.
# ----------------------------------------
echo "Deformable registration of baseline T1 to visit 2 T1..."
echo "Results stored in ${OUTDIR}/deformable/"
elastix -f $V2/nii/T1.nii -m $BASELINE/nii/T1.nii -p $PARAMETERDIR/Parameters_Rigid.txt -out $OUTDIR/deformable/ -t0 $OUTDIR/rigid/TransformParameters.0.txt