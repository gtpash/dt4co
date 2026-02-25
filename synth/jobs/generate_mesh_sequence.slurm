#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for generating sequence of meshes for sub-00101 on Frontera.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch generate_mesh_sequence.slurm" on a Frontera login node.
#
#   -- This script assumes that you have saved the sif image in $WORK/containers
#
#   -- This script assumes that you have saved the data in $SCRATCH/upenn_101_vbg/subjects/sub-00101/stl/
#
#----------------------------------------------------

#SBATCH -J generate_mesh_seq        # Job name
#SBATCH -o generate_mesh_seq.o.%j   # Name of stdout output file
#SBATCH -e generate_mesh_seq.e.%j   # Name of stderr error file
#SBATCH -p small                    # Queue (partition) name
#SBATCH -N 1                        # Total # of nodes 
#SBATCH -n 1                        # Total # of mpi tasks
#SBATCH -t 08:00:00                 # Run time (hh:mm:ss)
#SBATCH --mail-type=all             # Send email at begin and end of job
#SBATCH -A ASC24013                 # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=gtpash@utexas.edu

# Load the correct modules
module load mvapich2-x/2.3
module load tacc-apptainer
module list
date

MESH_LEVELS=( 32 64 96 128 192 256 )
SIF=$WORK/containers/onco-fenics_latest.sif
STLDIR=$SCRATCH/upenn_101_vbg/subjects/sub-00101/stl/
OUTDIR=$SCRATCH/sub-00101_mesh_convergence/sub-00101_mesh/

mkdir -p $OUTDIR
cd $OUTDIR
pwd

# generate the sequence of meshes
for L in "${MESH_LEVELS[@]}"
do
    echo "Generating mesh with resolution $L"
    MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 apptainer run $SIF python3 $DT4CO_PATH/gbm/meshing/mesh_generator.py \
        --stlpath $STLDIR \
        --outdir $OUTDIR \
        --resolution $L \
        --fname full$L \
        --hemi both
done
