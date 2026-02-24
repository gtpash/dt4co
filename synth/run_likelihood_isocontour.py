################################################################################
#
# This script is intended to evaluate the log-likelihood on a grid of parameter
# values, so that one can compute the isocontours of the log-likelihood function.
#
# An example call to this script is: #todo
# mpirun -np <num_procs> python3 run_likelihood_isocontour.py
#        --mesh /path/to/mesh/
#        --pdir /path/to/patient/data/
#        --imgfreq imaging_frequency
#        --experiment_type EXPERIMENT_TYPE
#        --samples /path/to/samples.h5
#        --nsamples num_samples
#        --sample_type SAMPLE_TYPE
#        --l2f
#        --pred_date prediction_date
#        --outdir /path/to/store/results/
#        --write_viz
#        -PETScOptions
#
# For more information run: python3 run_likelihood_isocontour.py --help
#
################################################################################

import os
import sys
import time
import argparse
import petsc4py  # before dolfin to avoid petsc4py.init error
import numpy as np


def main(args) -> None:
    # Load these modules here so that the petsc4py.init() call can handle the CLI args.
    import dolfin as dl

    sys.path.append(os.environ.get("HIPPYLIB_PATH"))
    import hippylib as hp

    sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
    from dt4co.synth import synthExperiment, setup_data_collection_timeline
    from dt4co.utils.mesh_utils import report_mesh_info, load_mesh
    from dt4co.utils.data_utils import nifti2Function
    from dt4co.utils.fenics_io import write_mv_to_h5, write_mv_to_xdmf, read_mv_from_h5
    from dt4co.utils.parallel import root_print

    # ------------------------------------------------------------
    # General setup.
    # ------------------------------------------------------------
    SEP = "\n" + "#" * 80 + "\n"

    dl.set_log_level(dl.LogLevel.WARNING)  # suppress FEniCS output.
    VERBOSE = args.verbose

    # MPI setup.
    COMM = dl.MPI.comm_world
    rank = COMM.rank

    root_print(COMM, SEP)

    # Paths for data.
    MESH_FPATH = args.mesh
    OUT_DIR = args.outdir
    PATIENT_DIR = args.pdir
    os.makedirs(OUT_DIR, exist_ok=True)  # output directory

    # set up the experiment
    EXP_TYPE = args.experiment_type
    NOISY_DATA = args.noisy

    IC_FILE = os.path.join(PATIENT_DIR, f"synthetic_obs_day_{int(0):03d}.nii")

    # ------------------------------------------------------------
    # Set up the experiment.
    # ------------------------------------------------------------

    IMG_FREQ = args.imgfreq  # imaging frequency [days] (every third day)
    TX_START = 14.0  # start of therapy [days]
    PRED_DATE = 14.0  # how long to predict after the last observation [days]
    N_HOLDOUT = 1  # number of holdout observations

    # Get the physical dimension, define mesh functions.
    mesh = load_mesh(COMM, MESH_FPATH)
    report_mesh_info(mesh)

    exp = synthExperiment()

    # ------------------------------------------------------------
    # Set up the function spaces.
    # ------------------------------------------------------------
    root_print(COMM, "Setting up function spaces and tissue segmentation indicator function.")
    Vh = exp.setupBIPFunctionSpaces(mesh, mle=False)
    assigner = exp.setupFunctionAssigner(mesh)
    mfun = dl.Function(Vh[hp.PARAMETER])

    # ------------------------------------------------------------
    # Set up the variational problem.
    # ------------------------------------------------------------
    root_print(COMM, "Setting up the forward model.")

    # set up a therapy timeline if using the therapy experiment.
    if EXP_TYPE == "rdtx":
        # set up the Stupp protocol for the patient.
        stupp_radio, stupp_chemo = exp.setupTXModels(tx_start=TX_START)

        # spoof the imaging timeline
        last_tx_day = max(stupp_radio.tx_days[-1], stupp_chemo.tx_days[-1])
        img_days = setup_data_collection_timeline(last_tx_day + 30.0, PRED_DATE, step=IMG_FREQ)  # go out to 1 month post-treatment
        t0 = img_days[0]
        tf = img_days[-1]
    else:
        # no therapy
        stupp_radio = None
        stupp_chemo = None

        img_days = setup_data_collection_timeline(60.0, PRED_DATE, step=IMG_FREQ)  # 2 months uncontrolled growth
        t0 = img_days[0]
        tf = img_days[-1]

    tf = img_days[-1 - N_HOLDOUT]
    root_print(COMM, "Set up the therapy models and imaging timeline.")

    # get the initial condition.
    u0 = dl.Function(Vh[hp.STATE])
    nifti2Function(IC_FILE, u0, Vh[hp.STATE])

    sparam = {"snes_view": None} if VERBOSE else None
    pde = exp.setupBIPVariationalProblem(Vh, u0, t0, tf, exptype=EXP_TYPE, sparams=sparam, radio_model=stupp_radio, chemo_model=stupp_chemo)
    root_print(COMM, "Set up the variational problem.")

    mprior = exp.setupPrior(Vh, mle=False)
    root_print(COMM, "Set up the prior.")

    # ------------------------------------------------------------
    # Spoof the misfits object.
    # ------------------------------------------------------------

    # set up the visit days for the misfit
    visit_days = img_days[1:-N_HOLDOUT]  # skip the first day, hold outs for misfit

    # build list of visits
    if NOISY_DATA:
        visits = [os.path.join(PATIENT_DIR, f"synthetic_noisy_obs_day_{int(day):03d}.nii") for day in visit_days]
    else:
        visits = [os.path.join(PATIENT_DIR, f"synthetic_obs_day_{int(day):03d}.nii") for day in visit_days]

    for i, date in enumerate(visit_days):
        root_print(COMM, f"Visit {i+1}: {date}")

    root_print(COMM, "Setting up the misfit object.")
    NOISE_VAR = exp.NOISE * exp.NOISE  # variance
    misfits = exp.spoofMisfitTD(visits=visits, visit_days=visit_days, Vh=Vh[hp.STATE], noise_var=NOISE_VAR, exnii=IC_FILE)

    # -----------------------------------------------------------
    # Read back the MAP point.
    # -----------------------------------------------------------

    root_print(COMM, SEP)
    root_print(COMM, f"Reading in the MAP point from file...")
    root_print(COMM, f"MAP file:\t{args.map}")

    # read back the MAP point.
    mmap = hp.MultiVector(mfun.vector(), 1)
    read_mv_from_h5(COMM, mmap, Vh[hp.PARAMETER], args.map, name=["map"])

    mapfun = dl.Function(Vh[hp.PARAMETER])
    mapfun.vector().axpy(1.0, mmap[0])  # copy the MAP point into a Function

    # -----------------------------------------------------------
    # Loop through steps and evaluate the likelihood.
    # -----------------------------------------------------------

    model = hp.Model(pde, mprior, misfits)  # set up the inverse problem
    steps = np.linspace(-1 * args.step, args.step, args.num_steps, endpoint=True)  # set up the mesh grid
    stepdir = dl.Function(Vh[hp.PARAMETER])

    llnp = np.zeros(len(steps), len(steps))  # array to store the log-likelihood values at each point in the grid

    # loop through the grid, evaluate the likelihood at each point.
    for ii, stepi in enumerate(steps):
        for jj, stepj in enumerate(steps):
            mfun.vector().zero()
            mfun.vector().axpy(1.0, mapfun.vector())  # start at the MAP point

            # step direction in the parameter space (log-space for D and K)
            stepdir.assign(dl.Constant([stepi * np.log(exp.D0), stepj * np.log(exp.K0)]))

            mfun.vector().axpy(1.0, stepdir.vector())  # take a step in the parameter space

            xx = model.generate_vector()  # generate a vector in the parameter space
            xx[hp.PARAMETER].axpy(1.0, mfun.vector())  # set the parameter vector to the current point in the grid

            # solve the forward problem, evaluate the misfit, compute the log-likelihood.
            model.solveFwd(xx[hp.STATE], xx)  # solve the forward problem to get the state at the current parameter point

            cost = model.cost(xx)  # evaluate the cost (negative log-likelihood) at the current point

            # todo: record location and cost value for visualization later.
            llnp[ii, jj] = cost[2]  # store only the misfit value

    if COMM.rank == 0:
        # save the log-likelihood values to file for later visualization.
        np.savez(os.path.join(args.outdir, "likelihood_isocontour_data.npz"), steps=steps, llnp=llnp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate log-likelihood on grid.")

    # data inputs.
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--pdir", type=str, required=True, help="Directory to where the (synthetic) patient data is stored.")
    parser.add_argument("--imgfreq", type=int, required=False, default=1, help="Frequency of images [days].")
    parser.add_argument("--noisy", action=argparse.BooleanOptionalAction, default=True, help="Whether or not to use measurements polluted with noise.")

    parser.add_argument("--map", type=str, required=True, help="File containing the MAP point.")

    # modeling inputs.
    parser.add_argument("--experiment_type", type=str, required=True, choices=["rd", "rdtx"], help="Type of experiment to run.")
    parser.add_argument("--num_steps", type=int, required=False, default=21, help="Number of samples in each dimension for evaluating the likelihood.")
    parser.add_argument("--step", type=float, required=False, default=0.1, help="Maximum step size for evaluating the likelihood.")

    # output options.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose output.")

    # Parse the arguments, strip CLI args for PETSc.
    args, other = parser.parse_known_args()
    petsc4py.init(other)

    main(args)
