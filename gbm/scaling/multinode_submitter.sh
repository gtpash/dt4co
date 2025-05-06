#!/bin/bash

##################################################################################################
# 
# This script will submit a slurm script for each number of nodes in the range [1, 32]
# to perform a strong scaling study across multiple nodes.
# 
# NOTE:
#   - Be sure to double check the SLURM_SCRIPT variable to ensure that the correct script is being submitted.
#   - Be sure to double check the directory in the DATADIR variable to ensure that the correct data is being processed.
# 
# Usage: bash multinode_submitter.sh
# 
##################################################################################################


# directory where patient data is stored.
QPLACEHOLDER="QUEUE"
NPLACEHOLDER="NNODES"
SLURM_SCRIPT="multinode_strong.slurm"

for i in {0..5}; do

    num_nodes=$((2**i))
    QUEUE="flex"

    # if [ ${num_nodes} -gt 3 ]; then
        # QUEUE="flex"
        # QUEUE="normal"
    # else
        # QUEUE="small"
    # fi

    # update the queue and number of nodes.
    sed -i -e "s/-p ${QPLACEHOLDER}/-p ${QUEUE}/g" ${SLURM_SCRIPT}
    sed -i -e "s/-N ${NPLACEHOLDER}/-N ${num_nodes}/g" ${SLURM_SCRIPT}
    sed -i -e "s/-n ${NPLACEHOLDER}/-n ${num_nodes}/g" ${SLURM_SCRIPT}  # to avoid slurm submission error

    # submit the script.
    sbatch ${SLURM_SCRIPT}

    # return the placeholders.
    sed -i -e "s/-p ${QUEUE}/-p ${QPLACEHOLDER}/g" ${SLURM_SCRIPT}
    sed -i -e "s/-N ${num_nodes}/-N ${NPLACEHOLDER}/g" ${SLURM_SCRIPT}
    sed -i -e "s/-n ${num_nodes}/-n ${NPLACEHOLDER}/g" ${SLURM_SCRIPT}  # to avoid slurm submission error

done
