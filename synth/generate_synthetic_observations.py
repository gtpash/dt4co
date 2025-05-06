################################################################################
# 
# This script generates synthetic observations for the synthetic experiment.
# 
# An example call to this script is:
# mpirun -np <num_procs> python3 generate_synthetic_observations.py
#       --pdir /path/to/patient/data/
#       --mesh /path/to/mesh/
#       --outdir /path/to/store/results/
#       --experiment_type EXPERIMENT_TYPE
# 
# For more information run: python3 generate_synthetic_observations.py --help
# 
# This code generates the following output:
#   - noisy synthetic observations in the $OUTDIR directory
#   - un-noised synthetic observations in the $OUTDIR directory
# 
################################################################################

import os
import sys
import time
import argparse
import petsc4py         # before dolfin to avoid petsc4py.init error
import numpy as np

def main(args)->None:
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
    SEP = "\n"+"#"*80+"\n"
    
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
    IMG_FREQ = 1        # imaging frequency [days] (every third day)
    TX_START = 14.0     # start of therapy [days]
    PRED_DATE = 30.0    # how long to predict after the last observation [days]
    
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
        
        # spoof the imaging timeline
        last_tx_day = max(stupp_radio.tx_days[-1], stupp_chemo.tx_days[-1])
        img_days = setup_data_collection_timeline(last_tx_day + 30.0, PRED_DATE, step=IMG_FREQ)    # go out to 1 month post-treatment
        t0 = img_days[0]
        tf = img_days[-1]
    else:
        # no therapy
        stupp_radio = None
        stupp_chemo = None
        
        img_days = setup_data_collection_timeline(60.0, PRED_DATE, step=IMG_FREQ)    # 2 months uncontrolled growth
        t0 = img_days[0]
        tf = img_days[-1]
    
    # get the initial condition.
    u0 = dl.Function(Vh[hp.STATE])
    nifti2Function(TUMOR_FILE, u0, Vh[hp.STATE])
    
    # if using the mollified tumor seed initial condition.
    if USE_TUMOR_SEED:
        RADIUS = 5.0    # tumor radius in cm
        VAL = 10.0      # needs to be tailored based on the tumor size
        xyz_com = computeFunctionCenterOfMass(u0, Vh[hp.STATE])
        u0moll = dl.project(MollifierInitialCondition(3, xyz_com, r=RADIUS, v=VAL), Vh[hp.STATE], solver_type="cg", preconditioner_type="jacobi")  # only for 3D
        u0.vector().zero()
        u0.vector().axpy(1., u0moll.vector())

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

    # pollute measurements with 2% noise
    max_state = x0[hp.STATE].norm("linf", "linf")
    noise_std_dev = exp.NOISE * max_state
    
    helpfun = dl.Function(Vh[hp.STATE])  # helper function for rasterization
    
    REF_NII = TUMOR_FILE
    obsOp = niftiPointwiseObservationOp(REF_NII, Vh[hp.STATE])  # observation operator
    
    img_days = np.arange(tf)  # image daily.
    for i, img_day in enumerate(img_days):
        
        RASTER_FILE = os.path.join(OUT_DIR, f"synthetic_obs_day_{int(img_day):03d}.nii")
        
        root_print(COMM, f"Rasterizing synthetic observation for day {img_day}.")
        root_print(COMM, f"Output file: {RASTER_FILE}")
        
        helpfun.vector().zero()
        helpfun.vector().axpy(1., x0[hp.STATE].view(img_day))
        rasterizeFunction(helpfun, Vh[hp.STATE], REF_NII, RASTER_FILE, obsOp=obsOp)
        
        NOISY_RASTER_FILE = os.path.join(OUT_DIR, f"synthetic_noisy_obs_day_{int(img_day):03d}.nii")
        root_print(COMM, f"Rasterizing noisy synthetic observation for day {img_day}.")
        root_print(COMM, f"Output file: {NOISY_RASTER_FILE}")
        noisyRasterizeFunction(helpfun, Vh[hp.STATE], REF_NII, NOISY_RASTER_FILE, obsOp=obsOp, noise_std_dev=noise_std_dev)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic observations for patient from true underlying PDE model.")
    
    # Required inputs.
    parser.add_argument("--pdir", type=str, help="Path to the patient data directory.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory to store observations.")
    parser.add_argument("--experiment_type", type=str, required=True, choices=["rd", "rdtx"], help="Type of experiment to run.")
    
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
