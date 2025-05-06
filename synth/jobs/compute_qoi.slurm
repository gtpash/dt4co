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
#   -- This script assumes that you have saved the data in $WORK/data/UPENN-GBM
#
#   -- This script assumes that the subject names are of the form "XXXXX"
#
#----------------------------------------------------

#SBATCH -J sub-00101_qoi                   # Job name
#SBATCH -o sub-00101_qoi.o.%j              # Name of stdout output file
#SBATCH -e sub-00101_qoi.e.%j              # Name of stderr error file
#SBATCH -p small                           # Queue (partition) name
#SBATCH -N 1                               # Total # of nodes 
#SBATCH -n 4                               # Total # of mpi tasks
#SBATCH -t 08:00:00                        # Run time (hh:mm:ss)
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
export IBRUN_TASKS_PER_NODE=56  # Set the number of tasks per node (56 is max for CLX node)
SIF=$SCRATCH/containers/onco-fenics_latest.sif

SUBID="00101"
EXPTYPE="rdtx"
IMGFREQ="1"
EXPNAME="${EXPTYPE}_freq${IMGFREQ}"

PDIR="$WORK/data/UPENN-GBM/sub-$SUBID/"
RESDIR="${SCRATCH}/hgg/sub-${SUBID}/"
BIPDIR="${RESDIR}/${EXPNAME}_bip/"
MESH="${RESDIR}/mesh/full64-all.h5"
SYNTHDIR="${RESDIR}/synthetic_${EXPTYPE}_obs/"

EVAL="$BIPDIR/${EXPNAME}_eigenvalues.txt"
EVEC="$BIPDIR/${EXPNAME}_eigenvectors_data.h5"
MAP="$BIPDIR/${EXPNAME}_param_data.h5"
NSAMPLES="500"

# setup for the QoI computations
THRESHOLD=0.1

PRED_DATE="28"
PRED_DATE=$(printf "%03d" $PRED_DATE)

PROP_DIR="$BIPDIR/fwd_prop/last_to_final_${PRED_DATE}"
# PROP_DIR="${BIPDIR}/fwd_prop/full/${SAMPLE_TYPE}"

REAL_PRED_DATE="113"
REAL_PRED_DATE=$(printf "%03d" $REAL_PRED_DATE)

#  ---------------------------------------------------------
# compute the QoIs for the prior
#  ---------------------------------------------------------

SAMPLE_TYPE="prior"
ALL_SAMPLES="${PROP_DIR}/${SAMPLE_TYPE}_day${REAL_PRED_DATE}_all_data.h5"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/compute_qoi.py --mesh $MESH --pdir $SYNTHDIR --samples $ALL_SAMPLES --sample_type $SAMPLE_TYPE --pred_date $REAL_PRED_DATE --threshold $THRESHOLD --outdir $PROP_DIR --nsamples $NSAMPLES

#  ---------------------------------------------------------
# compute the QoIs for the posterior
#  ---------------------------------------------------------

SAMPLE_TYPE="la_post"
ALL_SAMPLES="${PROP_DIR}/${SAMPLE_TYPE}_day${REAL_PRED_DATE}_all_data.h5"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/compute_qoi.py --mesh $MESH --pdir $SYNTHDIR --samples $ALL_SAMPLES --sample_type $SAMPLE_TYPE --pred_date $REAL_PRED_DATE --threshold $THRESHOLD --outdir $PROP_DIR --nsamples $NSAMPLES
