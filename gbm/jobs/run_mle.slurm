#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for running maximum likelihood estimation of the scalar coefficients.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_mle.slurm" on a Frontera login node.
#
#   -- This script assumes that you have saved the sif image in $SCRATCH/containers
#
#   -- This script assumes that you have saved the data in $WORK/data/IvyGAP
#
#   -- This script assumes that the subject names are of the form "WXX"
# 
#   -- This script assumes that the mesh has been generated and saved in $SCRATCH/hgg/sub-WXX/bip_mesh
# 
#   -- This script uses the pointwise observation operator and mollified reaction-diffusion model.
#
#----------------------------------------------------

#SBATCH -J sub-WXX_dmrd_mle             # Job name
#SBATCH -o sub-WXX_dmrd_mle.o.%j        # Name of stdout output file
#SBATCH -e sub-WXX_dmrd_mle.e.%j        # Name of stderr error file
#SBATCH -p small                        # Queue (partition) name
#SBATCH -N 2                            # Total # of nodes 
#SBATCH -n 2                            # Total # of mpi tasks
#SBATCH -t 04:00:00                     # Run time (hh:mm:ss)
#SBATCH --mail-type=all                 # Send email at begin and end of job
#SBATCH -A ASC24013                     # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=gtpash@utexas.edu

# Load the correct modules
module unload xalt
module load mvapich2-x/2.3
module load tacc-apptainer
module list
cd $SCRATCH
date
pwd

# set up the environment
SIF=$SCRATCH/containers/onco-fenics-2019.1.0_latest.sif  # path to Apptainer image
export IBRUN_TASKS_PER_NODE=56  # Set the number of tasks per node (56 is max for CLX node)

SUBID=WXX           # subject to be processed
EXPNAME=dmrd_mle    # experiment name
EXPTYPE=rd          # experiment type

# set up I/O
PDIR=$WORK/data/IvyGAP/$SUBID/
OUTDIR=$SCRATCH/hgg/sub-$SUBID/${EXPNAME}
MESH=$SCRATCH/hgg/sub-$SUBID/bip_mesh/bip_mesh.xdmf

# run the Bayesian inverse problem
MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/run_bip.py \
    --pinfo $PDIR/patient_info.json \
    --pdir $PDIR \
    --outdir $OUTDIR \
    --mesh $MESH \
    --pointwise \
    --prefix ${EXPNAME} \
    --experiment_type ${EXPTYPE} \
    --mle \
    --moll
