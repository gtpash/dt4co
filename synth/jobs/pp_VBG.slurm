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
#   -- This script assumes that you have saved the data in $WORK/data/UPENN-GBM
#
#   -- This script assumes that the subject names are of the form "XXXXX"
# 
#   -- This script assumes that the FreeSurfer subjects directory is $SCRATCH/subjects
#       This can be edited, but it must be set to overwrite the container default.
#
#----------------------------------------------------

#SBATCH -J pp_sub-XXXXX_vbg           # Job name
#SBATCH -o pp_sub-XXXXX_vbg.o.%j      # Name of stdout output file
#SBATCH -e pp_sub-XXXXX_vbg.e.%j      # Name of stderr error file
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
SUBID=00101

# set up the paths to the containers
SIF=$SCRATCH/containers/hgg-preprocess_latest.sif
PYSIF=$SCRATCH/containers/onco-fenics-2019.1.0_latest.sif
MESHSIF=$SCRATCH/containers/hgg-meshing_latest.sif  # non-compute version of Dolfin with ITK

# set up the paths to the data
DATADIR=$WORK/data/UPENN-GBM/sub-${SUBID}/
T1=$DATADIR/UPENN-GBM-${SUBID}_11_T1_unstripped.nii
OUTDIR=$SCRATCH/hgg/sub-$SUBID

mkdir -p $OUTDIR
cd $OUTDIR
pwd

# post-process to generate the STL files
apptainer run $SIF $DT4CO_PATH/gbm/preprocessing/postprocFreeSurfer.sh sub-$SUBID $T1 $SD

# generate a mesh
MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 apptainer run $PYSIF python3 $DT4CO_PATH/gbm/meshing/mesh_generator.py --stlpath $SD/sub-$SUBID/stl --outdir $OUTDIR/mesh/ --resolution 256 --fname full256 --hemi both

# registrations
MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 apptainer run $MESHSIF python3 $DT4CO_PATH/gbm/preprocessing/UPENN-GBM/reg_upenn_fs.py \
    --pdir $DATADIR \
    --elxparams $DT4CO_PATH/gbm/preprocessing/elastixParameters/ \
    --subdir $SUBJECTS_DIR \
     --subid $SUBID