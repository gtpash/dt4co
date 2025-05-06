#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for running the Bayesian inverse problem on the synthetic data.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_synth_rd_bip.slurm" on a Frontera login node.
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

#SBATCH -J sub-00101_rdtx_bip              # Job name
#SBATCH -o sub-00101_rdtx_bip.o.%j         # Name of stdout output file
#SBATCH -e sub-00101_rdtx_bip.e.%j         # Name of stderr error file
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
PDIR="$WORK/data/UPENN-GBM/sub-$SUBID/"
RESDIR="${SCRATCH}/hgg/sub-${SUBID}/"
OUTDIR="${RESDIR}/${EXPNAME}_bip/"
MESH="${RESDIR}/mesh/full64-all.h5"
SYNTHDIR="${RESDIR}/synthetic_${EXPTYPE}_obs/"

# run the Bayesian inverse problem
MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/run_synth_bip.py \
    --pdir $SYNTHDIR \
    --outdir $OUTDIR \
    --mesh $MESH \
    --imgfreq $IMGFREQ \
    --prefix ${EXPNAME} \
    --experiment_type $EXPTYPE \
    --noisy \
    -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg
