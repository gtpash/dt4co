#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for running VBG and FreeSurfer on Frontera.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_VBG.slurm" on a Frontera login node.
#
#   -- This script assumes that you have saved the sif image in $SCRATCH/containers
#
#   -- This script assumes that you have saved the data in $WORK/data/IvyGAP
#
#   -- This script assumes that the subject names are of the form "WXX"
#
#----------------------------------------------------

#SBATCH -J sub-WXX_vbg          # Job name
#SBATCH -o sub-WXX_vbg.o.%j     # Name of stdout output file
#SBATCH -e sub-WXX_vbg.e.%j     # Name of stderr error file
#SBATCH -p small                # Queue (partition) name
#SBATCH -N 1                    # Total # of nodes 
#SBATCH -n 1                    # Total # of mpi tasks
#SBATCH -t 12:00:00             # Run time (hh:mm:ss)
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH -A ASC24013             # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=gtpash@utexas.edu

# Load the correct modules
module load mvapich2-x/2.3
module load tacc-apptainer
module list
cd $SCRATCH
date
pwd

# --------------------------------------------------
# for IvyGAP dataset
# --------------------------------------------------
SUBID=WXX

# set up the environment
SIF=$SCRATCH/containers/hgg-preprocess_latest.sif
DATADIR=$WORK/data/IvyGAP/$SUBID/
T1=$DATADIR/T1_pre.nii
ROI=$DATADIR/ROI_nonenhance_v1.nii
T1_FILLED="sub-${SUBID}_T1_nat_filled.nii"

# --------------------------------------------------
# for UPENN-GBM dataset
# --------------------------------------------------
# SIF=$SCRATCH/containers/hgg-preprocess_latest.sif
# SUBID=00101
# DATADIR=$WORK/data/UPENN-GBM/sub-${SUBID}/
# T1=$DATADIR/UPENN-GBM-${SUBID}_11_T1_unstripped.nii
# ROI=$DATADIR/UPENN-GBM-${SUBID}_11_segm.nii
# T1_FILLED="sub-${SUBID}_T1_nat_filled.nii"

# gunzip the NIfTIs
gunzip $DATADIR/*.gz

# convert to RAS
apptainer run $SIF python3 $DT4CO_PATH/gbm/preprocessing/convert2RAS.py --dir $DATADIR

# run VBG
apptainer run $SIF KUL_VBG.sh -S $SUBID -a $T1 -z T1 -o $SCRATCH/vbg -l $ROI

# un-gunzip the filled image
gunzip $SCRATCH/vbg/output_VBG/sub-${SUBID}/${T1_FILLED}.gz

# run FreeSurfer
apptainer run $SIF recon-all -subject sub-$SUBID -i $SCRATCH/vbg/output_VBG/sub-${SUBID}/${T1_FILLED} -all -sd $SCRATCH/subjects
