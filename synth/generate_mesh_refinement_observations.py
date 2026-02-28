################################################################################
#
# This script generates synthetic observations for the mesh refinement study.
#
# For more information run: python3 generate_mesh_refinement_observations.py --help
#
# This code generates the following output:
#   - synthetic observation at the end of the first
#   - the support of the domain (for masking voxels in the domain)
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
    from dt4co.utils.mesh_utils import report_mesh_info, load_mesh_subs
    from dt4co.utils.data_utils import nifti2Function, niftiPointwiseObservationOp, rasterizeFunction, noisyRasterizeFunction
    from dt4co.utils.model_utils import solveIndicators, MollifierInitialCondition, computeFunctionCenterOfMass
    from dt4co.utils.fenics_io import write_mv_to_h5, write_mv_to_xdmf
    from dt4co.utils.parallel import root_print

    # ------------------------------------------------------------
    # Generic setup.
    # ------------------------------------------------------------
    # MPI setup.
    COMM = dl.MPI.comm_world
    SEP = "\n" + "#" * 80 + "\n"

    # Extract input arguments.
    VERBOSE = args.verbose
    MESH_FPATH = args.mesh
    PATIENT_DIR = args.pdir
    OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)  # output directory

    TUMOR_FILE = os.path.join(PATIENT_DIR, "tumor_fs.nii")
    USE_TUMOR_SEED = args.seed
    EXP_TYPE = args.experiment_type

    # Get the physical dimension, define mesh functions.
    mesh, subs, bndrys = load_mesh_subs(COMM, MESH_FPATH)
    report_mesh_info(mesh)

    # ------------------------------------------------------------
    # Set up the experiment.
    # ------------------------------------------------------------
    IMG_FREQ = 1  # imaging frequency [days] (every third day)
    TX_START = 14.0  # start of therapy [days]
    OBS_DATE = args.obs_date  # at what point should the observation be taken? (in days)

    exp = synthExperiment()

    # ------------------------------------------------------------
    # Set up the function spaces.
    # ------------------------------------------------------------
    root_print(COMM, "Setting up function spaces and tissue segmentation indicator function.")
    Vh = exp.setupFunctionSpaces(mesh)
    chi_gm = solveIndicators(mesh, subs, 1)

    # ------------------------------------------------------------
    # Set up the forward model.
    # ------------------------------------------------------------
    root_print(COMM, "Setting up the forward model.")

    # set up a therapy timeline if using the therapy experiment.
    if EXP_TYPE == "rdtx":
        # set up the Stupp protocol for the patient.
        stupp_radio, stupp_chemo = exp.setupTXModels(tx_start=TX_START)
    else:
        # no therapy
        stupp_radio = None
        stupp_chemo = None

    t0 = 0.0  # start time for forward solve
    tf = OBS_DATE  # end time for forward solve (when the observation is taken)
    exp.DELTA_T = args.dt  # time step for forward solve

    # get the initial condition.
    u0 = dl.Function(Vh[hp.STATE])
    nifti2Function(TUMOR_FILE, u0, Vh[hp.STATE])

    # if using the mollified tumor seed initial condition.
    if USE_TUMOR_SEED:
        RADIUS = 5.0  # tumor radius in cm
        VAL = 10.0  # needs to be tailored based on the tumor size
        xyz_com = computeFunctionCenterOfMass(u0, Vh[hp.STATE])
        u0moll = dl.project(MollifierInitialCondition(dim=3, center=xyz_com, r=RADIUS, v=VAL), Vh[hp.STATE], solver_type="cg", preconditioner_type="jacobi")  # only for 3D
        u0.vector().zero()
        u0.vector().axpy(1.0, u0moll.vector())

    sparam = {"snes_view": None} if VERBOSE else None
    pde = exp.setupVariationalProblem(Vh, u0, t0, tf, chi_gm=chi_gm, exptype=EXP_TYPE, radio_model=stupp_radio, chemo_model=stupp_chemo, moll=args.moll, sparam=sparam)

    # ------------------------------------------------------------
    # Set up the priors.
    # ------------------------------------------------------------

    root_print(COMM, "Setting up the priors.")

    # ------------------------------------------------------------
    # Run the model forward.
    # ------------------------------------------------------------
    utrue = pde.generate_state()
    mtrue = exp.trueParameter(Vh, sample=args.sample)

    write_mv_to_h5(COMM, [mtrue], Vh[hp.PARAMETER], os.path.join(OUT_DIR, "synthetic_true_parameter_data.h5"), name="true_parameter")
    write_mv_to_xdmf(COMM, [mtrue], Vh[hp.PARAMETER], os.path.join(OUT_DIR, "synthetic_true_parameter.xdmf"), name="true_parameter")

    x0 = [utrue, mtrue, None]

    root_print(COMM, "Beginning forward solve.")
    start = time.perf_counter()
    pde.solveFwd(x0[hp.STATE], x0)
    end = time.perf_counter() - start

    root_print(COMM, SEP)
    root_print(COMM, f"Forward solve took {end / 60:.2f} minutes.")
    root_print(COMM, SEP)

    pde.exportState(x0[hp.STATE], os.path.join(OUT_DIR, "synthetic_state.xdmf"))

    # ------------------------------------------------------------
    # Noisy measurements, rasterization.
    # ------------------------------------------------------------

    helpfun = dl.Function(Vh[hp.STATE])  # helper function for rasterization

    REF_NII = TUMOR_FILE
    obsOp = niftiPointwiseObservationOp(REF_NII, Vh[hp.STATE])  # observation operator

    # write out data
    root_print(COMM, "Rasterizing the support of the domain.")
    root_print(COMM, f"Output file: {os.path.join(OUT_DIR, 'domain_support.nii')}")
    support = dl.interpolate(dl.Constant(1.0), Vh[hp.STATE])  # support of the domain (for masking voxels in the domain)
    helpfun.vector().zero()
    helpfun.vector().axpy(1.0, support.vector())
    rasterizeFunction(helpfun, Vh[hp.STATE], REF_NII, os.path.join(OUT_DIR, "domain_support.nii"), obsOp=obsOp)

    root_print(COMM, f"Rasterizing the observations at at the final time {int(tf):03d}.")
    root_print(COMM, f"Output file: {os.path.join(OUT_DIR, f'observed.nii')}")
    helpfun.vector().zero()
    helpfun.vector().axpy(1.0, x0[hp.STATE].view(int(tf)))
    rasterizeFunction(helpfun, Vh[hp.STATE], REF_NII, os.path.join(OUT_DIR, f"observed.nii"), obsOp=obsOp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic observations for patient from true underlying PDE model for mesh refinement study.")

    # Required inputs.
    parser.add_argument("--pdir", type=str, help="Path to the patient data directory.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step to use for the forward solve (in days).")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory to store observations.")
    parser.add_argument("--experiment_type", type=str, required=True, choices=["rd", "rdtx"], help="Type of experiment to run.")

    parser.add_argument("--obs_date", type=float, default=28.0, help="Date for comparison observation (in days).")

    # Output options.
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose output.")

    # Input options.
    parser.add_argument("--moll", action=argparse.BooleanOptionalAction, default=False, help="Use mollified forward model?")
    parser.add_argument("--seed", action=argparse.BooleanOptionalAction, default=False, help="Use mollified (tumor seed) initial condition?")
    parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=False, help="Use a sample from the prior (instead of the mean)?")

    # Parse the arguments, strip CLI args for PETSc.
    args, other = parser.parse_known_args()
    petsc4py.init(other)

    main(args)
