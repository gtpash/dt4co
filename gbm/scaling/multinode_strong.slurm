#!/bin/bash
#----------------------------------------------------
# Slurm job to run strong scaling study on
#  multiple TACC Frontera CLX nodes
#
# Notes:
#
#  -- Launch this script by executing
#     "sbatch run_strong_scaling.slurm" on a Frontera login node.
#
#  -- This script assumes that you have saved the sif image in $SCRATCH
#
#  -- This script assumes that you have already run hIPPYlib at least once in the container.
# 
#  -- Use ibrun to launch MPI codes on TACC systems
#     Do NOT use mpirun or mpiexec.
#
#  -- This script attempts to reset the number of tasks on the node.
#     This may not work properly across multiple nodes.
#
#  -- The environment variables HIPPYLIB_PATH and DT4CO_PATH should be set in your .bashrc file
#
#----------------------------------------------------

#SBATCH -J multinode_strong_scaler          # Job name
#SBATCH -o multinode_strong_scaler.o.%j     # Name of stdout output file
#SBATCH -e multinode_strong_scaler.e.%j     # Name of stderr error file
#SBATCH -p QUEUE                            # Queue (partition) name
#SBATCH -N NNODES                           # Total # of nodes
#SBATCH -n NNODES                           # Total number of mpi tasks (start at 1)
#SBATCH -t 00:30:00                         # Run time (hh:mm:ss)
#SBATCH --mail-type=all                     # Send email at begin and end of job
#SBATCH -A ASC24013                         # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=gtpash@utexas.edu

# Load the correct modules
module load mvapich2-x/2.3
module load tacc-apptainer
module list
date
pwd

# Set the number of tasks per node
export IBRUN_TASKS_PER_NODE=56

# Set up paths for scaling study
OUTDIR=$DT4CO_PATH/gbm/scaling/logs/
LOG_FILE=multinode_tacc_strong.log
SIF=$SCRATCH/containers/onco-fenics_latest.sif
WORKSIZE=4000000

echo "Running strong scaling study..."
echo "Logs will be written to: $LOG_FILE"

echo "Running with slurm tasks per node: ${SLURM_TASKS_PER_NODE}"
echo "Running with number of slurm nodes: ${SLURM_NNODES}"
echo "Running with slurm procs: ${SLURM_NPROCS}"
echo "Running with slurm tasks: ${SLURM_NTASKS}"
echo "Running with ibrun tasks per node: ${IBRUN_TASKS_PER_NODE}"

# Solve the dang problem
MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 $DT4CO_PATH/gbm/scaling/strong_scaler.py --log $LOG_FILE --adjoint --work $WORKSIZE --outdir $OUTDIR
