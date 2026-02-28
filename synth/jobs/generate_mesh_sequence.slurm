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
#SBATCH -t 02:00:00                 # Run time (hh:mm:ss)
#SBATCH --mail-type=all             # Send email at begin and end of job
#SBATCH -A ASC24013                 # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=gtpash@utexas.edu

# Load the correct modules
module load mvapich2-x/2.3
module load tacc-apptainer
module list
date

SIF=$WORK/containers/onco-fenics_latest.sif
cd $SCRATCH
pwd

# ------------------------------------------------------------
# uncomment block to generate sequence of surface meshes (note that this is not uniform refinement!)
# ------------------------------------------------------------
# STLDIR=$SCRATCH/upenn_101_vbg/subjects/sub-00101/stl/
# OUTDIR=$SCRATCH/sub-00101/
# mkdir -p $OUTDIR

# MESH_LEVELS=( 32 64 128 256 )
# for L in "${MESH_LEVELS[@]}"
# do
#     echo "Generating mesh with resolution $L"
#     OUTDIR=$SCRATCH/sub-00101/mesh_${L}_refine_0
#     MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 apptainer run $SIF python3 $DT4CO_PATH/gbm/meshing/mesh_generator.py \
#         --stlpath $STLDIR \
#         --outdir $OUTDIR \
#         --resolution $L \
#         --fname full$L \
#         --hemi both
# done
# ------------------------------------------------------------

# generate sequence of uniformly refined meshes
REFINEMENT_LEVELS=( 1 2 3 )
BASE_RESOLUTION=32
MESH_DIR=$SCRATCH/sub-00101/

for L in "${REFINEMENT_LEVELS[@]}"
do
    echo "Generating mesh with refinement level $L"
    OUTDIR=$MESH_DIR/mesh_${BASE_RESOLUTION}_refine_$L
    mkdir -p ${OUTDIR}
    BASE_MESH=${MESH_DIR}/mesh_${BASE_RESOLUTION}_refine_$((L-1))/full${BASE_RESOLUTION}-all.h5
    MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 apptainer run $SIF python3 $DT4CO_PATH/gbm/meshing/refine_mesh.py \
        --mesh $BASE_MESH \
        --outdir $OUTDIR \
        --fname full${BASE_RESOLUTION}-all
done
