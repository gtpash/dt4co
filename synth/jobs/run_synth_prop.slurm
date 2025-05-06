#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for running the forward propagation for the synthetic data.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch run_synth_prop.slurm" on a Frontera login node.
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

#SBATCH -J sub-00101_prop              # Job name
#SBATCH -o sub-00101_prop.o.%j         # Name of stdout output file
#SBATCH -e sub-00101_prop.e.%j         # Name of stderr error file
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
NSAMPLES="1000"
DRAW_SAMPLES=false

#  ---------------------------------------------------------
# draw samples from the prior + LA posterior
#  ---------------------------------------------------------
if [ "$DRAW_SAMPLES" = true ] ; then
    MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/draw_la_samples.py --mesh $MESH --eval $EVAL --evec $EVEC --map $MAP --nsamples $NSAMPLES --outdir $BIPDIR --write_viz
fi

#  ---------------------------------------------------------
# setup for pushforward
#  ---------------------------------------------------------
PRED_DATE="28"
PRED_DATE=$(printf "%03d" $PRED_DATE)  # Pads with zero to "XXX" format
L2F_FLAG="--no-l2f"                    # Flag for last-to-final pushforward "--l2f" or "--no-l2f"
PROP_IMGFREQ="7"
VIZ_FLAG="--no-write_viz"              # Flag for writing visualization data "--write_viz" or "--no-write_viz"

#  ---------------------------------------------------------
# push forward the samples from the prior
#  ---------------------------------------------------------
OUTDIR="${BIPDIR}/fwd_prop"
SAMPLE_TYPE="prior"
SAMPLES="$BIPDIR/bip_${SAMPLE_TYPE}_samples_data.h5"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/run_fwd_prop.py --mesh $MESH --pdir $SYNTHDIR --imgfreq $PROP_IMGFREQ --no-noisy --experiment_type $EXPTYPE --samples $SAMPLES --nsamples $NSAMPLES --sample_type $SAMPLE_TYPE $L2F_FLAG --pred_date $PRED_DATE --outdir $OUTDIR $VIZ_FLAG -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg

# concatenate the results
if [ $L2F_FLAG == "--l2f" ]; then
    OUTDIR="${BIPDIR}/fwd_prop/last_to_final_${PRED_DATE}"
    DATADIR="${BIPDIR}/fwd_prop/last_to_final_${PRED_DATE}/${SAMPLE_TYPE}"
else
    OUTDIR="${BIPDIR}/fwd_prop/full/"
    DATADIR="${BIPDIR}/fwd_prop/full/${SAMPLE_TYPE}"
fi

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/concat_fwd_prop.py --mesh $MESH --datadir $DATADIR --imgfreq $PROP_IMGFREQ --experiment_type $EXPTYPE --outdir $OUTDIR $L2F_FLAG --pred_date $PRED_DATE --prefix $SAMPLE_TYPE

#  ---------------------------------------------------------
# push forward the prior mean
#  ---------------------------------------------------------
OUTDIR="${BIPDIR}/fwd_prop"
SAMPLE_TYPE="prior_mean"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/run_fwd_prop.py --mesh $MESH --pdir $SYNTHDIR --imgfreq $PROP_IMGFREQ --experiment_type $EXPTYPE --samples $MAP --nsamples 1 --sample_type $SAMPLE_TYPE $L2F_FLAG --pred_date $PRED_DATE --outdir $OUTDIR --write_viz -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg

#  ---------------------------------------------------------
# push forward the samples from the LA posterior
#  ---------------------------------------------------------
OUTDIR="${BIPDIR}/fwd_prop"
SAMPLE_TYPE="la_post"
SAMPLES="$BIPDIR/bip_${SAMPLE_TYPE}_samples_data.h5"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/run_fwd_prop.py --mesh $MESH --pdir $SYNTHDIR --imgfreq $PROP_IMGFREQ --no-noisy --experiment_type $EXPTYPE --samples $SAMPLES --nsamples $NSAMPLES --sample_type $SAMPLE_TYPE $L2F_FLAG --pred_date $PRED_DATE --outdir $OUTDIR $VIZ_FLAG -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg

# concatenate the results
if [ $L2F_FLAG == "--l2f" ]; then
    OUTDIR="${BIPDIR}/fwd_prop/last_to_final_${PRED_DATE}"
    DATADIR="${BIPDIR}/fwd_prop/last_to_final_${PRED_DATE}/${SAMPLE_TYPE}"
else
    OUTDIR="${BIPDIR}/fwd_prop/full/"
    DATADIR="${BIPDIR}/fwd_prop/full/${SAMPLE_TYPE}"
fi

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/concat_fwd_prop.py --mesh $MESH --datadir $DATADIR --imgfreq $PROP_IMGFREQ --experiment_type $EXPTYPE --outdir $OUTDIR $L2F_FLAG --pred_date $PRED_DATE --prefix $SAMPLE_TYPE

#  ---------------------------------------------------------
# pushforward the MAP point
#  ---------------------------------------------------------
OUTDIR="${BIPDIR}/fwd_prop"
SAMPLE_TYPE="map"

MV2_SMP_USE_CMA=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/run_fwd_prop.py --mesh $MESH --pdir $SYNTHDIR --imgfreq $PROP_IMGFREQ --experiment_type $EXPTYPE --samples $MAP --nsamples 1 --sample_type $SAMPLE_TYPE $L2F_FLAG --pred_date $PRED_DATE --outdir $OUTDIR --write_viz -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg
