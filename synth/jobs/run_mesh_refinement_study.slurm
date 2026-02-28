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
#SBATCH -N 4                        # Total # of nodes 
#SBATCH -n 4                        # Total # of mpi tasks
#SBATCH -t 04:00:00                 # Run time (hh:mm:ss)
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

REFINEMENT_LEVELS=( 0 1 2 3 )
TIMESTEPS=( 1 0.5 0.25 0.125)
OBS_DATE=28.0
BASE_RESOLUTION=32
MESH_DIR=$SCRATCH/sub-00101/mesh_refinement_study/

SIF=$WORK/containers/onco-fenics_latest.sif
PDIR=$WORK/data/UPENN-GBM/sub-00101/
STUDYDIR=$SCRATCH/sub-00101/mesh_refinement_study/

cd $SCRATCH
pwd

# generate data with the sequence of meshes
for L in "${REFINEMENT_LEVELS[@]}"
do
    DELTA_T=${TIMESTEPS[$L]}
    echo "Generating data with reslution $L and timestep ${DELTA_T}"
    MESHDIR=$STUDYDIR/mesh_${BASE_RESOLUTION}_refine_$L/
    MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 apptainer run $SIF python3 $DT4CO_PATH/synth/generate_synthetic_observations.py \
        --pdir $PDIR \
        --mesh $MESHDIR/full${BASE_RESOLUTION}-all.h5 \
        --outdir $MESHDIR \
        --experiment_type $EXPTYPE \
        --dt $DELTA_T \
        --obs_date $OBS_DATE \
        --verbose -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg
done
