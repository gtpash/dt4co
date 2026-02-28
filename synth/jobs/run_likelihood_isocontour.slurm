#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for evaluating the log-likelihood on a grid of parameter values, so that one can compute the isocontours of the log-likelihood function.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_likelihood_isocontour.slurm" on a Frontera login node.
#
#   -- This script assumes that you have saved the sif image in $SCRATCH/containers
#
#   -- This script assumes that you have saved the data in $WORK/data/UPENN-GBM
#
#   -- This script assumes that the subject names are of the form "XXXXX"
# 
#   -- This script uses the pointwise observation operator and reaction-diffusion model.
#
#----------------------------------------------------

#SBATCH -J sub-00101_likelihood_eval       # Job name
#SBATCH -o sub-00101_likelihood_eval.o.%j  # Name of stdout output file
#SBATCH -e sub-00101_likelihood_eval.e.%j  # Name of stderr error file
#SBATCH -p small                           # Queue (partition) name
#SBATCH -N 2                               # Total # of nodes 
#SBATCH -n 4                               # Total # of mpi tasks
#SBATCH -t 48:00:00                        # Run time (hh:mm:ss)
#SBATCH --mail-type=all                    # Send email at begin and end of job
#SBATCH -A ASC24013                        # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=gtpash@utexas.edu

# Load the correct modules
module unload xalt
module load tacc-apptainer
module load mvapich2-x/2.3
module list
cd $SCRATCH
date
pwd

# set up the environment
SIF=$SCRATCH/containers/onco-fenics_latest.sif  # path to Apptainer image
export IBRUN_TASKS_PER_NODE=56  # Set the number of tasks per node (56 is max for CLX node)

SUBID="00101"         # subject to be processed
EXPTYPE="rdtx"          # experiment type
IMGFREQ="1"           # frequency of observations [days]
EXPNAME="${EXPTYPE}_freq${IMGFREQ}"    # experiment name

# set up I/O
RESDIR="${SCRATCH}/sub-${SUBID}/"
OUTDIR="${RESDIR}/${EXPNAME}_bip/"
SYNTHDIR="${RESDIR}/synthetic_${EXPTYPE}_obs/"

BASE_RESOLUTION=64
REFINEMENT_LEVEL=0
MESH="${RESDIR}/mesh_${BASE_RESOLUTION}_refine_${REFINEMENT_LEVEL}/full${BASE_RESOLUTION}-all.h5"

BIPDIR="${RESDIR}/${EXPNAME}_bip/"
MAP="${BIPDIR}/${EXPNAME}_param_data.h5"

# set up the parameters for the sweep
STEP_SIZE=0.1  # maximum step size for the grid of parameter values to evaluate
NUM_STEPS=21   # number of steps in each direction

# run the likelihood evaluation on a grid of parameter values
MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/run_likelihood_isocontour.py \
    --pdir $SYNTHDIR \
    --outdir $OUTDIR \
    --mesh $MESH \
    --imgfreq $IMGFREQ \
    --prefix ${EXPNAME} \
    --experiment_type $EXPTYPE \
    --noisy \
    --map $MAP \
    --step $STEP_SIZE \
    --num_steps $NUM_STEPS \
    -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg
