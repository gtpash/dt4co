################################################################################
# 
# This script post-processes the single-node weak scaling study.
# 
# Usage: python3 pp_weak_scaling.py --log /path/to/logfile --outdir /path/to/output
# 
# Optionally, postprocess the strong scaling study for the adjoint solve
#   by suppling the argument --adjoint /path/to/adjoint/logfile
# 
################################################################################


import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt


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


def generate_weak_figure(fname, nproc, p_ws, obtained_ws):
    #----------------------------------------------------
    # Generate the figure
    #----------------------------------------------------
    
    p_ws_labels = [str(p) for p in p_ws]
    print("Generating figure...")

    fontsize=18

    plt.rc("figure", dpi=400)                   # High-quality figure ("dots-per-inch")
    plt.rc("text", usetex=True)                 # Crisp axis ticks
    plt.rc("font", family="sans-serif")         # Crisp axis labels
    plt.rc("legend", edgecolor='none')          # No boxes around legends
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams["figure.autolayout"] = True
    # rcParams['figure.constrained_layout.use'] = True

    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.2)

    charcoal    = [0.1, 0.1, 0.1]
    utblue = '#00A9B7'
    
    fig 	= plt.figure()
    ax11 	= fig.add_subplot(111)

    ax11.spines['right'].set_visible(False)
    ax11.spines['top'].set_visible(False)
    ax11.yaxis.set_ticks_position('left')
    ax11.xaxis.set_ticks_position('bottom')

    ##############
    p1 = ax11.semilogx(p_ws, np.ones_like(p_ws), linestyle='-', marker='o', lw=1.05, ms=3, color=charcoal)
    p2 = ax11.semilogx(p_ws, obtained_ws, linestyle='--', marker='*', lw=1.05, ms=5, color=utblue)

    # apply labels to figure.
    for i, p in enumerate(p_ws):
        if i >= 1:

            text = r'${{{}}}$'.format(truncate(obtained_ws[i]*100, 2))  + r'$\%$'		

            locX = 0.95
            locY = 0.97
            ax11.text(locX*p, locY*obtained_ws[i], text, color=utblue, rotation=0)

    ax11.set_ylabel('weak scaling efficiency')
    ax11.grid(True, linestyle='--', lw=0.3)

    ax11.set_ylim([0.6, 1.01])
    
    ax11.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax11.set_yticklabels([r'$60 \%$', r'$70 \%$', r'$80 \%$', r'$90 \%$', r'$100 \%$'])

    ax11.set_xlim([0.8, p_ws[-1] + 0.1*p_ws[-1]])
    ax11.set_xlabel('number of compute cores')
    ax11.set_xticks(p_ws)
    ax11.set_xticklabels(p_ws_labels)
    
    plt.savefig(fname)


def main(args)->None:
    
    os.makedirs(args.outdir, exist_ok=True)  # output directory
    
    print(f"Reading scaling data in from: {args.log}")
    data = get_data(args.log)
    
    nproc = data[:, 0].astype(int)          # number of processors
    p_ss = nproc
    wtime = data[:, -1]                     # wall-clock time

    obtained_ws = wtime[0] / wtime          # calculate the efficiency

    fname = f"{args.outdir}/weak_scaling.pdf"
    generate_weak_figure(fname, nproc, p_ss, obtained_ws)
    
    if args.adjoint is not None:
        print(f"Reading adjoint scaling data in from: {args.adjoint}")
        data = get_data(args.adjoint)
    
        nproc = data[:, 0].astype(int)          # number of processors
        p_ss = nproc
        wtime = data[:, -1]                     # wall-clock time

        obtained_ws = wtime[0] / wtime          # calculate the speedup

        fname = f"{args.outdir}/adjoint_weak_scaling.pdf"
        generate_weak_figure(fname, nproc, p_ss, obtained_ws)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocess scaling study.')

    parser.add_argument("--log", type=str, help="Log file of the scaling study.")
    parser.add_argument("--adjoint", type=str, default=None, help="Adjoint multinode log file.")
    parser.add_argument("--outdir", type=str, help="Where to save output.")

    args = parser.parse_args()

    main(args)
