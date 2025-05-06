#!/bin/bash

##################################################################################################
# 
# This script will submit a slurm script for each subject in the data directory.
# 
# NOTE:
#   - Be sure to double check the SLURM_SCRIPT variable to ensure that the correct script is being submitted.
# 
# Usage: bash prop_submitter.sh
# 
##################################################################################################


SLURM_SCRIPT="run_prop.slurm"

# inputs
PARTITION="posterior"
SUBID="W43"
SAMPLESPERJOB=30  # number of samples per job.
NJOBS=10  # number of jobs to submit.

# placeholders to be replaced in the slurm script.
SUB_PLACEHOLDER="WXX"
NSAMPLES_PLACEHOLDER="NUMSAMPLES"
SIDX_PLACEHOLDER="STARTIDX"
PARTITION_PLACEHOLDER="PARTITION"

for ((ii = 0; ii < $NJOBS; ++ii)); do
    
    SIDX=$((ii * $SAMPLESPERJOB))
    # replace the sample start.
    sed -i -e "s/${SUB_PLACEHOLDER}/${SUBID}/g" ${SLURM_SCRIPT}
    sed -i -e "s/SIDX=${SIDX_PLACEHOLDER}/SIDX=${SIDX}/g" ${SLURM_SCRIPT}
    sed -i -e "s/NSAMPLES=${NSAMPLES_PLACEHOLDER}/NSAMPLES=${SAMPLESPERJOB}/g" ${SLURM_SCRIPT}
    sed -i -e "s/PARTITION=${PARTITION_PLACEHOLDER}/PARTITION=${PARTITION}/g" ${SLURM_SCRIPT}

    # submit the script.
    sbatch ${SLURM_SCRIPT}

    # return the placeholder.
    sed -i -e "s/${SUBID}/${SUB_PLACEHOLDER}/g" ${SLURM_SCRIPT}
    sed -i -e "s/SIDX=${SIDX}/SIDX=${SIDX_PLACEHOLDER}/g" ${SLURM_SCRIPT}
    sed -i -e "s/NSAMPLES=${SAMPLESPERJOB}/NSAMPLES=${NSAMPLES_PLACEHOLDER}/g" ${SLURM_SCRIPT}
    sed -i -e "s/PARTITION=${PARTITION}/PARTITION=${PARTITION_PLACEHOLDER}/g" ${SLURM_SCRIPT}
done
