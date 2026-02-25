#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for generating synthetic observations for the mesh convergence study on Frontera.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_mesh_convergence_study.slurm" on a Frontera login node.
#
#   -- This script assumes that you have saved the sif image in $WORK/containers
#
#   -- This script assumes that you have previously generated the meshes and saved them in $SCRATCH/sub-00101_mesh_convergence/sub-00101_mesh/
#
#----------------------------------------------------

#SBATCH -J run_mesh_seq             # Job name
#SBATCH -o run_mesh_seq.o.%j        # Name of stdout output file
#SBATCH -e run_mesh_seq.e.%j        # Name of stderr error file
#SBATCH -p small                    # Queue (partition) name
#SBATCH -N 2                        # Total # of nodes 
#SBATCH -n 4                        # Total # of mpi tasks
#SBATCH -t 12:00:00                 # Run time (hh:mm:ss)
#SBATCH --mail-type=all             # Send email at begin and end of job
#SBATCH -A ASC24013                 # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=gtpash@utexas.edu

# Load the correct modules
module load mvapich2-x/2.3
module load tacc-apptainer
module list
date

export IBRUN_TASKS_PER_NODE=56  # Set the number of tasks per node (56 is max for CLX node)

EXPTYPE="rdtx"
MESH_LEVELS=( 32 64 96 128 192 256 )
SIF=$WORK/containers/onco-fenics_latest.sif
PDIR=$WORK/data/UPENN-GBM/sub-00101/
MESHDIR=$SCRATCH/sub-00101_mesh_convergence/sub-00101_mesh/
OUTDIR=$SCRATCH/sub-00101_mesh_convergence/

cd $OUTDIR
pwd

# generate the sequence of meshes
for L in "${MESH_LEVELS[@]}"
do
    echo "Generating mesh with resolution $L"
    MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 apptainer run $SIF python3 $DT4CO_PATH/synth/generate_synthetic_observations.py \
        --pdir $PDIR \
        --mesh $MESHDIR/full${L}-all.h5 \
        --outdir $OUTDIR/mesh_${L}_obs/ \
        --experiment_type $EXPTYPE \
        --verbose -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg
done
