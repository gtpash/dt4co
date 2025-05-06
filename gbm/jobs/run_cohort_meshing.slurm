#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for generating "production" meshes for IvyGAP cohort on Frontera.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_cohort_meshing.slurm" on a Frontera login node.
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

#SBATCH -J run_cohort_meshing       # Job name
#SBATCH -o run_cohort_meshing.o.%j  # Name of stdout output file
#SBATCH -e run_cohort_meshing.e.%j  # Name of stderr error file
#SBATCH -p small                    # Queue (partition) name
#SBATCH -N 1                        # Total # of nodes 
#SBATCH -n 1                        # Total # of mpi tasks
#SBATCH -t 04:00:00                 # Run time (hh:mm:ss)
#SBATCH --mail-type=all             # Send email at begin and end of job
#SBATCH -A ASC24013                 # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=gtpash@utexas.edu

# Load the correct modules
module load mvapich2-x/2.3
module load tacc-apptainer
module list
date

# overwrite the container default FreeSurfer subjects directory
SUBJECTS_DIR=$SCRATCH/subjects
SD=$SCRATCH/subjects

SIF=$SCRATCH/containers/onco-fenics-2019.1.0_latest.sif

# List of patients (Subjects)
PATIENTS=("W03" "W10" "W11" "W16" "W18" "W20" "W29" "W31" "W35" "W36" "W39" "W43" "W50" "W53")

# List of hemispheres (Primary Lesion Hemisphere)
HEMIS=("rh" "lh" "rh" "rh" "rh" "rh" "rh" "lh" "rh" "lh" "rh" "rh" "lh" "lh")

RESOLUTION=128

# Loop through both lists
for idx in "${!PATIENTS[@]}"
do
    patient=${PATIENTS[idx]}
    hemisphere=${HEMIS[idx]}

    # subject to be processed
    SUBID=${PATIENTS[idx]}

    # set up the paths to data
    DATADIR=$WORK/data/IvyGAP/$SUBID/
    T1=$DATADIR/T1_pre.nii

    OUTDIR=$SCRATCH/hgg/sub-$SUBID
    mkdir -p $OUTDIR

    # make sure to keep temporary files separated.
    cd $OUTDIR
    pwd
    
    # Tell the user what is being processed
    echo "Processing Patient: $patient, Primary Lesion Hemisphere: $hemisphere"

    # generate a mesh from the STL files.
    MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 apptainer run $SIF python3 $DT4CO_PATH/gbm/meshing/mesh_generator.py --stlpath $SD/sub-$SUBID/stl \
        --outdir $OUTDIR/bip_mesh/ --resolution $RESOLUTION --fname bip_mesh --hemi $hemisphere

done
