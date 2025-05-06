#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for running the Bayesian inverse problem.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_bip.slurm" on a Frontera login node.
#
#   -- This script assumes that you have saved the sif image in $SCRATCH/containers
#
#   -- This script assumes that you have saved the data in $WORK/data/IvyGAP
#
#   -- This script assumes that the subject names are of the form "WXX"
# 
#   -- This script assumes that the mesh has been generated and saved in $SCRATCH/hgg/sub-WXX/bip_mesh
# 
#   -- This script uses the pointwise observation operator and reaction-diffusion model.
#
#----------------------------------------------------

#SBATCH -J sub-WXX_dmpwrdtx_bip              # Job name
#SBATCH -o sub-WXX_dmpwrdtx_bip.o.%j         # Name of stdout output file
#SBATCH -e sub-WXX_dmpwrdtx_bip.e.%j         # Name of stderr error file
#SBATCH -p normal                            # Queue (partition) name
#SBATCH -N 4                                 # Total # of nodes 
#SBATCH -n 4                                 # Total # of mpi tasks
#SBATCH -t 48:00:00                          # Run time (hh:mm:ss)
#SBATCH --mail-type=all                      # Send email at begin and end of job
#SBATCH -A ASC24013                          # Project/Allocation name (req'd if you have more than 1)
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
SIF=$SCRATCH/containers/onco-fenics_latest.sif  # path to Apptainer image
export IBRUN_TASKS_PER_NODE=56  # Set the number of tasks per node (56 is max for CLX node)

# subject to be processed
SUBID="WXX"
EXPTYPE="pwrdtx"
EXPNAME="dm${EXPTYPE}"
DRAW_SAMPLES=true
NSAMPLES="500"

# set up I/O
PDIR="$WORK/data/IvyGAP/$SUBID/"
BIPDIR="$SCRATCH/hgg/sub-$SUBID/${EXPNAME}_bip/"
MESH="$SCRATCH/hgg/sub-$SUBID/bip_mesh/bip_mesh-all.h5"
PROPDIR="$SCRATCH/hgg/sub-$SUBID/${EXPNAME}_prop/"

# run the Bayesian inverse problem
MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/run_bip.py \
    --pinfo $PDIR/patient_info.json \
    --pdir $PDIR \
    --outdir $BIPDIR \
    --mesh $MESH \
    --prefix $EXPNAME \
    --experiment_type $EXPTYPE \
    --pointwise \
    --moll


# generate samples for prior / LA posterior pushforwards

EVAL="$BIPDIR/${EXPNAME}_eigenvalues.txt"
EVEC="$BIPDIR/${EXPNAME}_eigenvectors_data.h5"
MAP="$BIPDIR/${EXPNAME}_param_data.h5"

#  ---------------------------------------------------------
# draw samples from the prior + LA posterior
#  ---------------------------------------------------------
if [ "$DRAW_SAMPLES" = true ] ; then
    MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/draw_la_samples.py --pinfo $PDIR/patient_info.json --pdir $PDIR --experiment_type $EXPTYPE --eval $EVAL --evec $EVEC --map $MAP --mesh $MESH --nsamples $NSAMPLES --outdir $BIPDIR --prefix $EXPNAME --write_viz
fi
