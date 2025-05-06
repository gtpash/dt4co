#!/bin/bash
#----------------------------------------------------
# Slurm job to run strong scaling study on
#  TACC Frontera CLX nodes
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

#SBATCH -J strong_scaler        # Job name
#SBATCH -o strong_scaler.o.%j   # Name of stdout output file
#SBATCH -e strong_scaler.e.%j   # Name of stderr error file
#SBATCH -p small                # Queue (partition) name
#SBATCH -N 1                    # Total # of nodes
#SBATCH -n 1                    # Total number of mpi tasks (start at 1)
#SBATCH -t 04:00:00             # Run time (hh:mm:ss)
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH -A ASC24013             # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=gtpash@utexas.edu

# Load the correct modules
module load mvapich2-x/2.3
module load tacc-apptainer
module list
date
pwd

OUTDIR=$DT4CO_PATH/gbm/scaling/logs/
LOG_FILE="tacc_strong.log"
SIF="$SCRATCH/containers/onco-fenics_latest.sif"
WORKSIZE=1000000
# note: OOM on one core with 2MM vertices

echo "Running strong scaling study..."
echo "Logs will be written to: $LOG_FILE"

for i in {0..5};
do
   echo "Running with $((2**i)) processes..."
   export IBRUN_TASKS_PER_NODE=$((2**i))
   MV2_SMP_USE_CMA=0 ibrun -np $((2**i)) apptainer run $SIF python3 strong_scaler.py --log $LOG_FILE --adjoint --work $WORKSIZE --outdir $OUTDIR
done

# max out the CLX node
export IBRUN_TASKS_PER_NODE=56
MV2_SMP_USE_CMA=0 ibrun -np 56 apptainer run $SIF python3 strong_scaler.py --log $LOG_FILE --adjoint --work $WORKSIZE --outdir $OUTDIR
