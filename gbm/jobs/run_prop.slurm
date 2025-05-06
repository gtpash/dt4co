#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for running the forward propagation code
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_prop.slurm" on a Frontera login node.
#
#   -- This script assumes that you have saved the sif image in $SCRATCH/containers
#
#   -- This script assumes that you have saved the data as
#    posterior_samples_data.h5 and prior_samples_data.h5
#    in the $SCRATCH/hgg/sub-WXX/EXPNAME_prop/ directory
#
#   -- This script assumes that the subject names are of the form "WXX"
# 
#   -- This script assumes that the mesh has been generated and saved in $SCRATCH/hgg/sub-WXX/bip_mesh
# 
#   -- Be sure to update the EXPTYPE / EXPNAME variables as needed.
#
#----------------------------------------------------

#SBATCH -J sub-WXX_dmrd_prop                  # Job name
#SBATCH -o sub-WXX_dmrd_prop.o.%j             # Name of stdout output file
#SBATCH -e sub-WXX_dmrd_prop.e.%j             # Name of stderr error file
#SBATCH -p normal                         # Queue (partition) name
#SBATCH -N 4                             # Total # of nodes 
#SBATCH -n 4                             # Total # of mpi tasks
#SBATCH -t 18:00:00                      # Run time (hh:mm:ss)
#SBATCH --mail-type=all                  # Send email at begin and end of job
#SBATCH -A ASC24013                      # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=gtpash@utexas.edu

# Load the correct modules
module load mvapich2-x/2.3
module load tacc-apptainer
module list
cd $SCRATCH
date
pwd

# set up the environment
SIF=$SCRATCH/containers/onco-fenics-2019.1.0_latest.sif  # path to Apptainer image
export IBRUN_TASKS_PER_NODE=56  # Set the number of tasks per node (56 is max for CLX node)

# Set up information.
SUBID="WXX"
NSAMPLES=NUMSAMPLES
SIDX=STARTIDX
EXPNAME=dmrd
PARTITION=PARTITION
EXPTYPE=rd  # experiment type

# set up parameters
if [ $PARTITION == "posterior" ]; then
    SAMPLESPATH=$SCRATCH/hgg/sub-$SUBID/${EXPNAME}_prop/${EXPNAME}_posterior_samples_data.h5
    DATANAME=post_sample
    PREFIX="post"
elif [ $PARTITION == "prior" ]; then
    SAMPLESPATH=$SCRATCH/hgg/sub-$SUBID/${EXPNAME}_prop/${EXPNAME}_prior_samples_data.h5
    DATANAME=prior_sample
    PREFIX="prior"
else
    echo "Invalid partition name"
    exit 1
fi

# set up I/O
PDIR=$WORK/data/IvyGAP/$SUBID/
OUTDIR=$SCRATCH/hgg/sub-$SUBID/${EXPNAME}_prop/$PARTITION
MESH=$SCRATCH/hgg/sub-$SUBID/bip_mesh/bip_mesh-all.h5

# run the forward propagation code
MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/run_rd_forward.py \
    --pinfo $PDIR/patient_info.json \
    --pdir $PDIR \
    --mesh $MESH \
    --experiment_type ${EXPTYPE} \
    --samples $SAMPLESPATH \
    --dataname $DATANAME \
    --nsamples $NSAMPLES \
    --outdir $OUTDIR \
    --datadir $DATADIR \
    --uq \
    --sidx $SIDX \
    --prefix $PREFIX
