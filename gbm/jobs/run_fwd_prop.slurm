#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for running the forward propagation.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_fwd_prop.slurm" on a Frontera login node.
#
#   -- This script assumes that you have saved the sif image in $SCRATCH/containers
#
#   -- This script assumes that the subject names are of the form "XXXXX"
# 
#   -- This script uses the pointwise observation operator and reaction-diffusion model.
#
#----------------------------------------------------

#SBATCH -J sub-WXX_pwrdtx_prop                  # Job name
#SBATCH -o sub-WXX_pwrdtx_prop.o.%j             # Name of stdout output file
#SBATCH -e sub-WXX_pwrdtx_prop.e.%j             # Name of stderr error file
#SBATCH -p normal                         # Queue (partition) name
#SBATCH -N 4                             # Total # of nodes 
#SBATCH -n 4                             # Total # of mpi tasks
#SBATCH -t 36:00:00                      # Run time (hh:mm:ss)
#SBATCH --mail-type=all                  # Send email at begin and end of job
#SBATCH -A AST24009                      # Project/Allocation name (req'd if you have more than 1)
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
DRAW_SAMPLES=false

# set up for the pushforwards
L2F_FLAG="--l2f"                    # Flag for last-to-final pushforward "--l2f" or "--no-l2f"
VIZ_FLAG="--no-write_viz"                # Flag for writing visualization data "--write_viz" or "--no-write_viz"

#  ---------------------------------------------------------
# draw samples from the prior + LA posterior
#  ---------------------------------------------------------
if [ "$DRAW_SAMPLES" = true ] ; then
    MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/draw_la_samples.py --pinfo $PDIR/patient_info.json --pdir $PDIR --experiment_type $EXPTYPE --eval $EVAL --evec $EVEC --map $MAP --mesh $MESH --nsamples $NSAMPLES --outdir $BIPDIR --prefix $EXPNAME --write_viz
fi

#  ---------------------------------------------------------
# push forward the samples from the prior
#  ---------------------------------------------------------
OUTDIR="${BIPDIR}/fwd_prop"
SAMPLE_TYPE="prior"
SAMPLES="$BIPDIR/${EXPNAME}_${SAMPLE_TYPE}_samples_data.h5"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/run_fwd_prop.py --mesh $MESH --pdir $PDIR --pinfo $PDIR/patient_info.json --experiment_type $EXPTYPE --samples $SAMPLES --nsamples $NSAMPLES --sample_type $SAMPLE_TYPE --outdir $OUTDIR $VIZ_FLAG $L2F_FLAG -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg

# concatenate the results
if [ $L2F_FLAG == "--l2f" ]; then
    OUTDIR="${BIPDIR}/fwd_prop/last_to_final"
    DATADIR="${BIPDIR}/fwd_prop/last_to_final/${SAMPLE_TYPE}"
else
    OUTDIR="${BIPDIR}/fwd_prop/full/"
    DATADIR="${BIPDIR}/fwd_prop/full/${SAMPLE_TYPE}"
fi

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/concat_fwd_prop.py --mesh $MESH --pdir $PDIR --pinfo $PDIR/patient_info.json --datadir $DATADIR --experiment_type $EXPTYPE --outdir $OUTDIR $L2F_FLAG --prefix $SAMPLE_TYPE

#  ---------------------------------------------------------
# push forward the prior mean
#  ---------------------------------------------------------
OUTDIR="${BIPDIR}/fwd_prop"
SAMPLE_TYPE="prior_mean"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/run_fwd_prop.py --mesh $MESH --pdir $PDIR --pinfo $PDIR/patient_info.json --experiment_type $EXPTYPE --samples $MAP --nsamples $NSAMPLES --sample_type $SAMPLE_TYPE --outdir $OUTDIR $VIZ_FLAG $L2F_FLAG -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg

#  ---------------------------------------------------------
# push forward the samples from the LA posterior
#  ---------------------------------------------------------
OUTDIR="${BIPDIR}/fwd_prop"
SAMPLE_TYPE="la_post"
SAMPLES="$BIPDIR/${EXPNAME}_${SAMPLE_TYPE}_samples_data.h5"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/run_fwd_prop.py --mesh $MESH --pdir $PDIR --pinfo $PDIR/patient_info.json --experiment_type $EXPTYPE --samples $SAMPLES --nsamples $NSAMPLES --sample_type $SAMPLE_TYPE --outdir $OUTDIR $VIZ_FLAG $L2F_FLAG -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg

# concatenate the results
if [ $L2F_FLAG == "--l2f" ]; then
    OUTDIR="${BIPDIR}/fwd_prop/last_to_final"
    DATADIR="${BIPDIR}/fwd_prop/last_to_final/${SAMPLE_TYPE}"
else
    OUTDIR="${BIPDIR}/fwd_prop/full/"
    DATADIR="${BIPDIR}/fwd_prop/full/${SAMPLE_TYPE}"
fi

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/concat_fwd_prop.py --mesh $MESH --pdir $PDIR --pinfo $PDIR/patient_info.json --datadir $DATADIR --experiment_type $EXPTYPE --outdir $OUTDIR $L2F_FLAG --prefix $SAMPLE_TYPE

#  ---------------------------------------------------------
# pushforward the MAP point
#  ---------------------------------------------------------
OUTDIR="${BIPDIR}/fwd_prop"
SAMPLE_TYPE="map"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/gbm/run_fwd_prop.py --mesh $MESH --pdir $PDIR --pinfo $PDIR/patient_info.json --experiment_type $EXPTYPE --samples $MAP --nsamples $NSAMPLES --sample_type $SAMPLE_TYPE --outdir $OUTDIR $VIZ_FLAG $L2F_FLAG -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg
