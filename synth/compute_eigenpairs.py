################################################################################
#
# This script is intended to compute the eigenpairs of the prior-preconditioned
# Hessian at the MAP point for a given synthetic patient.
#
# For more information run: python3 compute_eigenpairs.py --help
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
    from dt4co.utils.fenics_io import read_mv_from_h5
    from dt4co.utils.parallel import root_print

    # ------------------------------------------------------------
    # General setup.
    # ------------------------------------------------------------
    SEP = "\n" + "#" * 80 + "\n"

    dl.set_log_level(dl.LogLevel.WARNING)  # suppress FEniCS output.
    VERBOSE = args.verbose

    # MPI setup.
    COMM = dl.MPI.comm_world
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
    # Read back the MAP point, set up linearization point.
    # -----------------------------------------------------------

    root_print(COMM, SEP)
    root_print(COMM, f"Reading in the MAP point from file...")
    root_print(COMM, f"MAP file:\t{args.map}")

    # read back the MAP point.
    mmap = hp.MultiVector(mfun.vector(), 1)
    read_mv_from_h5(COMM, mmap, Vh[hp.PARAMETER], args.map, name=["map"])

    mapfun = dl.Function(Vh[hp.PARAMETER])
    mapfun.vector().axpy(1.0, mmap[0])  # copy the MAP point into a Function

    model = hp.Model(pde, mprior, misfits)  # set up the inverse problem

    x = model.generate_vector()
    x[hp.PARAMETER].axpy(1.0, mapfun.vector())  # set the parameter vector to the MAP point
    model.solveFwd(x[hp.STATE], x)  # solve the forward problem at the MAP point to get the state

    # ------------------------------------------------------------
    # Compute the spectrum.
    # ------------------------------------------------------------

    kk = args.num_evals  # number of requested eigenvalues
    pp = args.oversample  # oversampling parameter for the double pass algorithm
    root_print(COMM, f"Double Pass Algorithm. Requested eigenvectors: {kk}, oversampling {pp}.")

    # Set the linearization point.
    model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
    Hmisfit = hp.ReducedHessian(model, misfit_only=True)

    Omega = hp.MultiVector(x[hp.PARAMETER], kk + pp)
    hp.parRandom.normal(1.0, Omega)
    d, U = hp.doublePassG(Hmisfit, mprior.R, mprior.Rsolver, Omega, kk, s=1, check=False)

    if COMM.rank == 0:
        np.savetxt(os.path.joint(args.outdir, f"eigenvalues_{args.num_evals}.txt"), d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute prior-preconditioned Hessian spectrum.")

    # data inputs.
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--pdir", type=str, required=True, help="Directory to where the (synthetic) patient data is stored.")
    parser.add_argument("--imgfreq", type=int, required=False, default=1, help="Frequency of images [days].")
    parser.add_argument("--noisy", action=argparse.BooleanOptionalAction, default=True, help="Whether or not to use measurements polluted with noise.")

    parser.add_argument("--map", type=str, required=True, help="File containing the MAP point.")

    # modeling inputs.
    parser.add_argument("--experiment_type", type=str, required=True, choices=["rd", "rdtx"], help="Type of experiment to run.")
    parser.add_argument("--num_evals", type=int, default=512, help="Number of eigenpairs to compute.")
    parser.add_argument("--oversample", type=int, default=10, help="Oversampling parameter for the double pass algorithm.")

    # output options.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose output.")

    # Parse the arguments, strip CLI args for PETSc.
    args, other = parser.parse_known_args()
    petsc4py.init(other)

    main(args)
