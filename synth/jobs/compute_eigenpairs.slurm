#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for computing eigenpairs of prior-preconditioned Hessian on Frontera.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_mesh_convergence_study.slurm" on a Frontera login node.
#
#   -- This script assumes that you have saved the sif image in $WORK/containers
#
#   -- This script assumes that you have previously computed the MAP point.
#
#----------------------------------------------------

#SBATCH -J compute_eig              # Job name
#SBATCH -o compute_eieg.o.%j        # Name of stdout output file
#SBATCH -e compute_eieg.e.%j        # Name of stderr error file
#SBATCH -p small                    # Queue (partition) name
#SBATCH -N 4                        # Total # of nodes 
#SBATCH -n 4                        # Total # of mpi tasks
#SBATCH -t 48:00:00                 # Run time (hh:mm:ss)
#SBATCH --mail-type=all             # Send email at begin and end of job
#SBATCH -A ASC24013                 # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=gtpash@utexas.edu

# Load the correct modules
module load mvapich2-x/2.3
module load tacc-apptainer
module list
date

export IBRUN_TASKS_PER_NODE=56  # Set the number of tasks per node (56 is max for CLX node)
SIF=$WORK/containers/onco-fenics_latest.sif
PDIR=$WORK/data/UPENN-GBM/sub-00101/

NUM_EVALS=512
OVERSAMPLE=16

EXPTYPE="rdtx"
IMGFREQ="1"           # frequency of observations [days]
SUBID="00101"         # subject to be processed
EXPNAME="${EXPTYPE}_freq${IMGFREQ}"    # experiment name

BIPDIR=$SCRATCH/sub-00101_study/${EXPNAME}_bip/
# BIPDIR=$SCRATCH/sub-00101_study/${EXPNAME}_bip_refined/
SYNTHDIR=$SCRATCH/sub-00101_study/synthetic_${EXPTYPE}_obs/
MESH=$SCRATCH/sub-00101_study/mesh/full64-all.h5
# MESH=$SCRATCH/sub-00101_study/mesh/full128-all.h5
MAP=$BIPDIR/${EXPNAME}_param_data.h5

cd $OUTDIR
pwd

MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/compute_eigenpairs.py \
    --pdir $PDIR \
    --mesh $MESH \
    --outdir $BIPDIR \
    --imgfreq $IMGFREQ \
    --experiment_type $EXPTYPE \
    --num_evals $NUM_EVALS \
    --oversample $OVERSAMPLE \
    --noisy \
    -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg
