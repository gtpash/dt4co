#!/bin/bash

##################################################################################################
# This script converts DICOM files to NIfTI files using dcm2niix.
# 
# The resultant NIfTI files are saved in the "nii" directory of the input data path.
# The DICOM data are assumed to be organized by imaging modality in subdirectories.
# Accepted modalities are: T1, T2, T2FLAIR, T1C, ADC, DWI.
# 
# Usage: ./convertDCM2NII.sh <data path>
# 
##################################################################################################

if [[ $# -eq 0 ]] ; then
    echo "ERROR: No data path provided."
    exit 1
fi

# this script converts DICOM files to NIFTI files using dcm2niix.
PARENTDIR=${1}
echo "converting DICOM files in ${PARENTDIR} to NIfTI..."

# ensure that a directory exists for the output.
NIFTIDIR="${PARENTDIR}/nii"
mkdir -p $NIFTIDIR

# declare an array of imaging modalities to be converted.
declare -a modalities=("T1" "T2" "T2FLAIR" "T1C" "ADC" "DWI")

for img in "${modalities[@]}"
do
    echo "converting ${img} DICOM to NIfTI..."
    
    # ensure that the DICOM data exists
    if [ -d "${PARENTDIR}/${img}" ];
    then
        if [ -f "${NIFTIDIR}/${img}.nii" ];
        then
            echo "WARNING: ${img} NIfTI data already exists. Skipping..."
        else
            dcm2niix -f ${img} -o "${NIFTIDIR}" "${PARENTDIR}/${img}"
        fi
    else
        echo "ERROR: ${modality} DICOM data not found. Skipping..."
    fi
done
