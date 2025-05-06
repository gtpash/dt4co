#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for computing QoIs after the forward propagation.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch compute_qoi.slurm" on a Frontera login node.
#
#   -- This script assumes that you have saved the sif image in $SCRATCH/containers
#
#   -- This script assumes that you have saved the data in $WORK/data/IvyGAP
#
#   -- This script assumes that the subject names are of the form "WXX"
# 
#   -- Be sure to appropriately update the number of visits for full trajectories
# 
#----------------------------------------------------

#SBATCH -J sub-WXX_qoi                   # Job name
#SBATCH -o sub-WXX_qoi.o.%j              # Name of stdout output file
#SBATCH -e sub-WXX_qoi.e.%j              # Name of stderr error file
#SBATCH -p small                           # Queue (partition) name
#SBATCH -N 1                               # Total # of nodes 
#SBATCH -n 4                               # Total # of mpi tasks
#SBATCH -t 04:00:00                        # Run time (hh:mm:ss)
#SBATCH --mail-type=all                    # Send email at begin and end of job
#SBATCH -A AST24009                        # Project/Allocation name (req'd if you have more than 1)
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
export IBRUN_TASKS_PER_NODE=56  # Set the number of tasks per node (56 is max for CLX node)
SIF=$SCRATCH/containers/onco-fenics_latest.sif

SUBID="WXX"
NVISIT="ZZ"
EXPTYPE="pwrdtx"
EXPNAME="dm${EXPTYPE}"

PDIR="$WORK/data/IvyGAP/${SUBID}/"
RESDIR="${SCRATCH}/hgg/sub-${SUBID}/"
BIPDIR="${RESDIR}/${EXPNAME}_bip/"
MESH="${RESDIR}/bip_mesh/bip_mesh-all.h5"

EVAL="$BIPDIR/${EXPNAME}_eigenvalues.txt"
EVEC="$BIPDIR/${EXPNAME}_eigenvectors_data.h5"
MAP="$BIPDIR/${EXPNAME}_param_data.h5"
NSAMPLES="500"

# setup for the QoI computations
THRESHOLD=0.1
VOX_FLAG="--no-vox"  # Flag for computing the QoIs on the voxel data "--vox" or "--no-vox"
L2F_FLAG="--no-l2f"

if [ $L2F_FLAG == "--l2f" ]; then
    PROP_DIR="${BIPDIR}/fwd_prop/last_to_final/"
    VIDX="v0"
else
    PROP_DIR="${BIPDIR}/fwd_prop/full/"
    VIDX="v$NVISIT"
fi

#  ---------------------------------------------------------
# compute the QoIs for the prior
#  ---------------------------------------------------------

SAMPLE_TYPE="prior"
ALL_SAMPLES="${PROP_DIR}/${SAMPLE_TYPE}_{$VIDX}_all_data.h5"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/compute_qoi.py --mesh $MESH --pdir $PDIR --pinfo $PDIR/patient_info.json --experiment_type $EXPTYPE --samples $ALL_SAMPLES --sample_type $SAMPLE_TYPE --threshold $THRESHOLD --outdir $PROP_DIR --nsamples $NSAMPLES $VOX_FLAG

#  ---------------------------------------------------------
# compute the QoIs for the posterior
#  ---------------------------------------------------------

SAMPLE_TYPE="la_post"
ALL_SAMPLES="${PROP_DIR}/${SAMPLE_TYPE}_{$VIDX}_all_data.h5"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/compute_qoi.py --mesh $MESH --pdir $PDIR --pinfo $PDIR/patient_info.json --experiment_type $EXPTYPE --samples $ALL_SAMPLES --sample_type $SAMPLE_TYPE --threshold $THRESHOLD --outdir $PROP_DIR --nsamples $NSAMPLES $VOX_FLAG

#  ---------------------------------------------------------
# compute the QoIs for the prior mean
#  ---------------------------------------------------------
SAMPLE_TYPE="prior_mean"
ALL_SAMPLES="${PROP_DIR}/${SAMPLE_TYPE}/${SAMPLE_TYPE}_000000_data.h5"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/compute_qoi.py --mesh $MESH --pdir $PDIR --pinfo $PDIR/patient_info.json --experiment_type $EXPTYPE --samples $ALL_SAMPLES --sample_type $SAMPLE_TYPE --threshold $THRESHOLD --outdir $PROP_DIR --nsamples 1 $VOX_FLAG

#  ---------------------------------------------------------
# compute the QoIs for the MAP point
#  ---------------------------------------------------------
SAMPLE_TYPE="map"
ALL_SAMPLES="${PROP_DIR}/${SAMPLE_TYPE}/${SAMPLE_TYPE}_000000_data.h5"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/compute_qoi.py --mesh $MESH --pdir $PDIR --pinfo $PDIR/patient_info.json --experiment_type $EXPTYPE --samples $ALL_SAMPLES --sample_type $SAMPLE_TYPE --threshold $THRESHOLD --outdir $PROP_DIR --nsamples 1 $VOX_FLAG

