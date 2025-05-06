#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for post-processing VBG + FreeSurfer results on Frontera.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch pp_VBG.slurm" on a Frontera login node.
#
#   -- This script assumes that you have saved the sif image in $SCRATCH/containers
#
#   -- This script assumes that you have saved the data in $WORK/data/IvyGAP
#
#   -- This script assumes that the subject names are of the form "WXX"
# 
#   -- This script assumes that the FreeSurfer subjects directory is $SCRATCH/subjects
#       This can be edited, but it must be set to overwrite the container default.
#
#----------------------------------------------------

#SBATCH -J pp_sub-WXX_vbg           # Job name
#SBATCH -o pp_sub-WXX_vbg.o.%j      # Name of stdout output file
#SBATCH -e pp_sub-WXX_vbg.e.%j      # Name of stderr error file
#SBATCH -p small                    # Queue (partition) name
#SBATCH -N 1                        # Total # of nodes 
#SBATCH -n 1                        # Total # of mpi tasks
#SBATCH -t 00:30:00                 # Run time (hh:mm:ss)
#SBATCH --mail-type=all             # Send email at begin and end of job
#SBATCH -A ASC24013                 # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=gtpash@utexas.edu

# Load the correct modules
module load mvapich2-x/2.3
module load tacc-apptainer
module list
date

# overwrite the container default
SUBJECTS_DIR=$SCRATCH/subjects
SD=$SCRATCH/subjects

# subject to be processed
SUBID=WXX

# set up the paths to the containers
SIF=$SCRATCH/containers/hgg-preprocess_latest.sif

PYSIF=$SCRATCH/containers/onco-fenics-2019.1.0_latest.sif

# non-compute version of Dolfin with ITK
ITKSIF=$SCRATCH/containers/hgg-meshing_latest.sif

# set up the paths to data
DATADIR=$WORK/data/IvyGAP/$SUBID/
T1=$DATADIR/T1_pre.nii

OUTDIR=$SCRATCH/hgg/sub-$SUBID
mkdir -p $OUTDIR

# make sure to keep temporary files separated.
cd $OUTDIR
pwd

# run the post-processing script to generate STL files
apptainer run $SIF $DT4CO_PATH/gbm/preprocessing/postprocFreeSurfer.sh sub-$SUBID $T1 $SD

# generate a mesh from the STL files.
MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 apptainer run $PYSIF python3 $DT4CO_PATH/gbm/meshing/mesh_generator.py --stlpath $SD/sub-$SUBID/stl \
    --outdir $OUTDIR/mesh/ --resolution 64 --fname full64 --hemi both

# build the patient information object.
MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 apptainer run --env SUBJECTS_DIR=${SUBJECTS_DIR} $ITKSIF python3 $DT4CO_PATH/gbm/preprocessing/IvyGAP/pp_manual.py \
    --pdir $DATADIR \
    --pinfo $DATADIR/$SUBID.json \
    --elxparams $DT4CO_PATH/gbm/preprocessing/elastixParameters \
    --subjid sub-$SUBID

# run the visualization script.
MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 apptainer run $PYSIF python3 $DT4CO_PATH/gbm/viz_registration.py \
    --pinfo $DATADIR/patient_info.json --datadir $DATADIR \
    --mesh $OUTDIR/mesh/full64.xdmf --outdir $OUTDIR/viz
