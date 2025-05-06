#!/bin/bash

#----------------------------------------------------
# 
# Slurm job for generating the synthetic datasets.
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch generate_synth_data.slurm" on a Frontera login node.
#
#   -- This script assumes that you have saved the sif image in $SCRATCH/containers
#
#   -- This script assumes that you have saved the data in $WORK/data/UPENN-GBM
#
#   -- This script assumes that the subject names are of the form "XXXXX"
#
#----------------------------------------------------

#SBATCH -J gen_synth_data                  # Job name
#SBATCH -o gen_synth_data.o.%j             # Name of stdout output file
#SBATCH -e gen_synth_data.e.%j             # Name of stderr error file
#SBATCH -p flex                           # Queue (partition) name
#SBATCH -N 2                               # Total # of nodes 
#SBATCH -n 4                               # Total # of mpi tasks
#SBATCH -t 04:00:00                        # Run time (hh:mm:ss)
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
SIF=$SCRATCH/containers/onco-fenics_latest.sif  # path to Apptainer image
export IBRUN_TASKS_PER_NODE=56  # Set the number of tasks per node (56 is max for CLX node)

# set up paths to common directories.
SUBID="00101"
PDIR="$WORK/data/UPENN-GBM/sub-$SUBID/"
RESDIR="${SCRATCH}/hgg/sub-${SUBID}"
MESH="${RESDIR}/mesh/full128-all.h5"

# ----------------------------------------------------
# No therapy. Mollified IC.
# ----------------------------------------------------
# EXPTYPE="rd"
# OUTDIR="${RESDIR}/synthetic_seed_${EXPTYPE}_obs/"

# echo "Generating synthetic data for subject $SUBID with experiment type $EXPTYPE"
# echo "Using mollified IC"
# echo "Output directory: $OUTDIR"

# MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/generate_synthetic_observations.py \
#     --pdir $PDIR \
#     --mesh $MESH \
#     --outdir $OUTDIR \
#     --experiment_type $EXPTYPE \
#     --seed \
#     --verbose -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg

# ----------------------------------------------------
# No therapy. True IC.
# ----------------------------------------------------
# EXPTYPE="rd"
# OUTDIR="${RESDIR}/synthetic_${EXPTYPE}_obs/"

# echo "Generating synthetic data for subject $SUBID with experiment type $EXPTYPE"
# echo "Using true IC"
# echo "Output directory: $OUTDIR"

# MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/generate_synthetic_observations.py \
#     --pdir $PDIR \
#     --mesh $MESH \
#     --outdir $OUTDIR \
#     --experiment_type $EXPTYPE \
#     --verbose -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg

# ----------------------------------------------------
# Including therapy. Mollified IC.
# ----------------------------------------------------
# EXPTYPE="rdtx"
# OUTDIR="${RESDIR}/synthetic_seed_${EXPTYPE}_obs/"

# echo "Generating synthetic data for subject $SUBID with experiment type $EXPTYPE"
# echo "Using mollified IC"
# echo "Output directory: $OUTDIR"

# MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/generate_synthetic_observations.py \
#     --pdir $PDIR \
#     --mesh $MESH \
#     --outdir $OUTDIR \
#     --experiment_type $EXPTYPE \
#     --seed \
#     --verbose -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg

# ----------------------------------------------------
# Including therapy. True IC.
# ----------------------------------------------------
EXPTYPE="rdtx"
OUTDIR="${RESDIR}/synthetic_${EXPTYPE}_obs/"

echo "Generating synthetic data for subject $SUBID with experiment type $EXPTYPE"
echo "Using true IC"
echo "Output directory: $OUTDIR"

MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/generate_synthetic_observations.py \
    --pdir $PDIR \
    --mesh $MESH \
    --outdir $OUTDIR \
    --experiment_type $EXPTYPE \
    --verbose -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg

# ----------------------------------------------------
# Including therapy. True IC. Sampled true parameter.
# ----------------------------------------------------
# EXPTYPE="rdtx"
# OUTDIR="${RESDIR}/synthetic_sampled_${EXPTYPE}_obs/"

# echo "Generating synthetic data for subject $SUBID with experiment type $EXPTYPE"
# echo "Using true IC"
# echo "Using sampled true parameter"
# echo "Output directory: $OUTDIR"

# MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/generate_synthetic_observations.py \
#     --pdir $PDIR \
#     --mesh $MESH \
#     --outdir $OUTDIR \
#     --experiment_type $EXPTYPE \
#     --verbose --sample -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg

# ----------------------------------------------------
# No therapy. True IC. Sampled true parameter.
# ----------------------------------------------------
# EXPTYPE="rd"
# OUTDIR="${RESDIR}/synthetic_sampled_${EXPTYPE}_obs/"

# echo "Generating synthetic data for subject $SUBID with experiment type $EXPTYPE"
# echo "Using true IC"
# echo "Using sampled true parameter"
# echo "Output directory: $OUTDIR"

# MV2_SMP_USE_CMA=0 MV2_ENABLE_AFFINITY=0 ibrun apptainer run $SIF python3 -u $DT4CO_PATH/synth/generate_synthetic_observations.py \
#     --pdir $PDIR \
#     --mesh $MESH \
#     --outdir $OUTDIR \
#     --experiment_type $EXPTYPE \
#     --verbose --sample -fwd_ksp_type gmres -fwd_pc_type hypre -fwd_pc_hypre_type boomeramg
