################################################################################
# 
# This script post-processes the multi-node strong scaling study.
# 
# Usage: python3 pp_multinode.py --log /path/to/logfile --outdir /path/to/output
# 
# Optionally, postprocess the strong scaling study for the adjoint solve
#   by suppling the argument --adjoint /path/to/adjoint/logfile
# 
################################################################################

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

def truncate(number, digits) -> float:
    # Improve accuracy with floating point operations, to avoid truncate(16.4, 2) = 16.39 or truncate(-1.13, 2) = -1.12
    nbDecimals = len(str(number).split('.')[1]) 
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits

    return math.trunc(stepper * number) / stepper


def get_data(file):

    data = np.loadtxt(file)             # load data from log file
    arr = data[np.argsort(data[:, 0])]  # sort by the number of processors
    
    return arr


def generate_strong_figure(fname, nproc, p_ss, ideal_ss, obtained_ss):
    #----------------------------------------------------
    # Generate the figure
    #----------------------------------------------------
    
    p_ss_labels = [str(p) for p in p_ss]
    print("Generating figure...")

    fontsize=14

    plt.rc("figure", dpi=400)                   # High-quality figure ("dots-per-inch")
    plt.rc("text", usetex=True)                 # Crisp axis ticks
    plt.rc("font", family="sans-serif")         # Crisp axis labels
    plt.rc("legend", edgecolor='none')          # No boxes around legends
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    plt.rcParams["figure.figsize"] = (6, 3)
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams["figure.autolayout"] = True
    # rcParams['figure.constrained_layout.use'] = True

    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.2)

    charcoal    = [0.1, 0.1, 0.1]
    utorange = '#BF5700'
    
    fig 	= plt.figure()
    ax11 	= fig.add_subplot(111)

    ax11.spines['right'].set_visible(False)
    ax11.spines['top'].set_visible(False)
    ax11.yaxis.set_ticks_position('left')
    ax11.xaxis.set_ticks_position('bottom')

    ##############
    p1 = ax11.loglog(p_ss, ideal_ss, linestyle='-', marker='o', lw=1.1, ms=3, color=charcoal)
    p2 = ax11.loglog(p_ss, obtained_ss, linestyle='--', marker='*', lw=1.1, ms=5, color=utorange)

    Y_OFFSET_OBTAINED = 0.6 #0.8
    Y_OFFSET_IDEAL = 1.1 #1.05
    X_OFFSET_OBTAINED = 0.9
    X_OFFSET_IDEAL = 0.9

    for i, p in enumerate(p_ss):
        
        if i != 0:
            ax11.text(X_OFFSET_IDEAL*p, ideal_ss[i]*Y_OFFSET_IDEAL, ideal_ss[i], color=charcoal, rotation=20)
            ax11.text(X_OFFSET_OBTAINED*p, obtained_ss[i]*Y_OFFSET_OBTAINED, truncate(obtained_ss[i], 2), color=utorange, rotation=20)
    
    MIDPOINT = np.median(ideal_ss) * nproc[0]  # number of processors at the midpoint of the plot
    
    ax11.text(MIDPOINT, 11, 'ideal strong scaling', color=charcoal, rotation=20)
    ax11.text(MIDPOINT, 1.5, 'obtained strong scaling', color=utorange, rotation=20)
    
    # for adjoint
    # ax11.text(MIDPOINT, 9, 'ideal strong scaling', color=charcoal, rotation=20)
    # ax11.text(MIDPOINT, 1, 'obtained strong scaling', color=utorange, rotation=20)


    #ax11.set_xlabel('number of processing units p')
    ax11.set_xticks(p_ss)
    ax11.set_xticklabels(p_ss_labels)

    ax11.set_ylabel('speed-up')
    ax11.grid(True, linestyle='--', lw=0.3)

    ax11.set_xlim([p_ss[0] - 0.1, p_ss[-1] + 0.1*p_ss[-1]])
    ax11.set_ylim([ideal_ss[0] - 0.2, ideal_ss[-1] + 0.2*ideal_ss[-1]])

    ax11.set_yticks(ideal_ss)
    ax11.set_yticklabels(ideal_ss)
    
    ax11.set_xlabel('number of compute cores')
    
    plt.savefig(fname, bbox_inches='tight')


def main(args):
    
    os.makedirs(args.outdir, exist_ok=True)  # output directory
    
    print(f"Reading scaling data in from: {args.log}")
    
    data = get_data(args.log)
    
    nproc = data[:, 0].astype(int)          # number of processors
    p_ss = nproc
    wtime = data[:, -1]                     # wall-clock time

    obtained_ss = wtime[0] / wtime          # calculate the speedup
    ideal_ss = nproc // nproc[0]            # calculate the speedup

    fname = f"{args.outdir}/multinode_strong_scaling.pdf"
    generate_strong_figure(fname, nproc, p_ss, ideal_ss, obtained_ss)
    
    if args.adjoint is not None:
        print(f"Reading adjoint scaling data in from: {args.adjoint}")
        data = get_data(args.adjoint)
    
        nproc = data[:, 0].astype(int)          # number of processors
        p_ss = nproc
        wtime = data[:, -1]                     # wall-clock time

        obtained_ss = wtime[0] / wtime          # calculate the speedup
        ideal_ss = nproc // nproc[0]            # calculate the speedup

        fname = f"{args.outdir}/multinode_adjoint_strong_scaling.pdf"
        generate_strong_figure(fname, nproc, p_ss, ideal_ss, obtained_ss)
    
    ###################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Postprocess scaling study.')

    parser.add_argument("--log", type=str, help="Log file of the scaling study.")
    parser.add_argument("--adjoint", type=str, default=None, help="Adjoint multinode log file.")
    parser.add_argument("--outdir", type=str, help="Where to save output.")

    args = parser.parse_args()

    main(args)
