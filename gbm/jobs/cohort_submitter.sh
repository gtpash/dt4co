#!/bin/bash

##################################################################################################
# 
# This script will submit a slurm script for each subject in the data directory.
# 
# NOTE:
# 
#   -- Be sure to double check the SLURM_SCRIPT variable to ensure that the correct script is being submitted.
# 
#   -- Currently the cohort is hardcoded, but can be modified to read from a directory.
# 
#   -- Be sure to double check the directory in the DATADIR variable to ensure that the correct data is being processed.
# 
# Usage: bash cohort_submitter.sh
# 
##################################################################################################

# Define the cohort of subjects.
# COHORT=("W03" "W10" "W11" "W16" "W18" "W20" "W29" "W31" "W35" "W36" "W39" "W43" "W50" "W53")
COHORT=("W11" "W16" "W18" "W29" "W31" "W35" "W36" "W39" "W50" "W53")

# directory where patient data is stored.
# DATADIR="${WORK}/data/IvyGAP/"

PLACEHOLDER="WXX"

# SLURM_SCRIPT="run_VBG.slurm"
# SLURM_SCRIPT="pp_VBG.slurm"
# SLURM_SCRIPT="run_rd_mle.slurm"
# SLURM_SCRIPT="run_bip.slurm"
SLURM_SCRIPT="run_fwd_prop.slurm"


# for dir in ${DATADIR}/*/; do  # loop through each subject directory.
for idx in "${!COHORT[@]}"; do

  # SUBID=$(basename $dir)  # get the subject ID from the directory name.
  SUBID=${COHORT[$idx]}

  # replace the subject ID.
  sed -i -e "s/${PLACEHOLDER}/${SUBID}/g" ${SLURM_SCRIPT}

  # submit the script.
  sbatch ${SLURM_SCRIPT}

  # return the placeholder.
  sed -i -e "s/${SUBID}/${PLACEHOLDER}/g" ${SLURM_SCRIPT}

done

