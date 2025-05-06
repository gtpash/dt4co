################################################################################
# 
# This script is intended to compute the quanities of interest at a single visit
# given the state data from the forward propagation.
# 
# An example call to this script is:
# mpirun -np <num_procs> python3 run_fwd_prop.py
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
# For more information run: python3 run_fwd_prop.py --help
# 
################################################################################

import os
import sys
import time
import argparse
import petsc4py         # before dolfin to avoid petsc4py.init error

def main(args)->None:
    # Load these modules here so that the petsc4py.init() call can handle the CLI args.
    import dolfin as dl
    
    sys.path.append(os.environ.get("HIPPYLIB_PATH"))
    import hippylib as hp
    
    sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
    from dt4co.synth import synthExperiment, setup_data_collection_timeline
    from dt4co.utils.mesh_utils import report_mesh_info, load_mesh
    from dt4co.utils.data_utils import nifti2Function
    from dt4co.utils.fenics_io import write_mv_to_h5, write_mv_to_xdmf, read_mv_from_h5, getGroupSize
    from dt4co.utils.parallel import root_print
    
    # ------------------------------------------------------------
    # General setup.
    # ------------------------------------------------------------
    SEP = "\n"+"#"*80+"\n"
    SUFFIX = f"_data"  # for the HDF5 data files.
    
    dl.set_log_level(dl.LogLevel.WARNING)  # suppress FEniCS output.
    VERBOSE = args.verbose
    
    # MPI setup.
    COMM = dl.MPI.comm_world
    rank = COMM.rank
    nproc = COMM.size
    
    root_print(COMM, SEP)
    
    # Paths for data.
    MESH_FPATH = args.mesh
    OUT_DIR = args.outdir
    PATIENT_DIR = args.pdir
    
    EXP_TYPE = args.experiment_type
    IS_L2F = args.l2f
    SAMPLE_TYPE = args.sample_type
    NSAMPLES = args.nsamples
    WRITE_VIZ = args.write_viz

    IMG_FREQ = args.imgfreq     # imaging frequency [days] (every third day)
    IMG_FREQ = 1 if IS_L2F else IMG_FREQ  # set up proper last observation if doing last-to-final
    PRED_DATE = args.pred_date  # how long to predict after the last observation [days]
    TX_START = 14.0             # start of therapy [days]
    N_HOLDOUT = 1               # number of holdout observations

    SAMPLES_FILE = args.samples
    if SAMPLE_TYPE == "map" or SAMPLE_TYPE == "prior_mean":
        SAMPLE_NAME = [SAMPLE_TYPE]
        if NSAMPLES != 1:
            NSAMPLES = 1
            root_print(COMM, f"Setting NSAMPLES to 1 for use with type: {SAMPLE_TYPE}.")
    else:
        SAMPLE_NAME = f"{SAMPLE_TYPE}_sample"
        NSAMPLES = NSAMPLES if NSAMPLES > 0 else getGroupSize(SAMPLES_FILE, SAMPLE_NAME)

    # set up directories to store output data
    STATE_DIR = os.path.join(OUT_DIR, f"last_to_final_{PRED_DATE:03d}") if IS_L2F else os.path.join(OUT_DIR, "full")
    PROP_DIR = os.path.join(STATE_DIR, SAMPLE_TYPE)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(PROP_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # Set up the experiment.
    # ------------------------------------------------------------
    
    # Get the physical dimension, define mesh functions.
    mesh = load_mesh(COMM, MESH_FPATH)
    report_mesh_info(mesh)
    
    exp = synthExperiment()
    
    root_print(COMM, "Setting the function spaces.")
    Vh = exp.setupBIPFunctionSpaces(mesh, mle=False)
    
    root_print(COMM, "Setting up the forward model.")
    # set up a therapy timeline if using the therapy experiment.
    if EXP_TYPE == "rdtx":
        # set up the Stupp protocol for the patient.
        stupp_radio, stupp_chemo = exp.setupTXModels(tx_start=TX_START)
        
        # spoof the imaging timeline
        last_tx_day = max(stupp_radio.tx_days[-1], stupp_chemo.tx_days[-1])
        img_days = setup_data_collection_timeline(last_tx_day + 30.0, PRED_DATE, step=IMG_FREQ)    # go out to 1 month post-treatment
    else:
        # no therapy
        stupp_radio = None
        stupp_chemo = None
        
        img_days = setup_data_collection_timeline(60.0, PRED_DATE, step=IMG_FREQ)    # 2 months uncontrolled growth
    
    t0 = img_days[-1 - N_HOLDOUT] if IS_L2F else img_days[0]
    tf = img_days[-1]
    
    visit_days = [img_days[-1]] if IS_L2F else img_days
    
    root_print(COMM, f"Initial time: {t0:.2f}")
    root_print(COMM, f"Final time: {tf:.2f}")
    
    # set the IC from file based on the determined initial time.
    IC_FILE = os.path.join(PATIENT_DIR, f"synthetic_obs_day_{int(t0):03d}.nii")
    
    u0 = dl.Function(Vh[hp.STATE])
    nifti2Function(IC_FILE, u0, Vh[hp.STATE])

    sparam = {"snes_view": None} if VERBOSE else None
    pde = exp.setupBIPVariationalProblem(Vh, u0, t0, tf, exptype=EXP_TYPE, sparams=sparam, radio_model=stupp_radio, chemo_model=stupp_chemo)
    
    # ------------------------------------------------------------
    # Load the sample data from file.
    # ------------------------------------------------------------
    
    root_print(COMM, SEP)
    root_print(COMM, "Reading back samples.")
    root_print(COMM, f"Samples are of type: {SAMPLE_TYPE}")
    root_print(COMM, f"Number of samples: {NSAMPLES}")
    
    # multi-vectors to store the samples.
    mmv = hp.MultiVector(pde.generate_parameter(), NSAMPLES)
    read_mv_from_h5(COMM, mmv, Vh[hp.PARAMETER], SAMPLES_FILE, name=SAMPLE_NAME)
    
    # ------------------------------------------------------------
    # Run the model forward.
    # ------------------------------------------------------------
    
    root_print(COMM, SEP)
    root_print(COMM, "Pushing forward the samples.")
    root_print(COMM, SEP)

    for ii in range(mmv.nvec()):
        m0 = mmv[ii]  # extract sample from multivector
        
        root_print(COMM, f"Beginning the forward solve for parameter sample: {ii}")
        root_print(COMM, f"Solve {ii+1}/{mmv.nvec()} for this run.")
        root_print(COMM, SEP)
    
        # list to store solution, parameter, and adjoint.
        x0 = [pde.generate_state(), m0, None]
        
        # paths to the data
        STATE_VIZ = os.path.join(PROP_DIR, f"{SAMPLE_TYPE}_state{ii:06d}.xdmf")
        STATE_DATA = os.path.join(PROP_DIR, f"{SAMPLE_TYPE}_{ii:06d}{SUFFIX}.h5")
        
        start = time.perf_counter()
        try:
            pde.solveFwd(x0[hp.STATE], x0)
        except:
            root_print(COMM, f"Error in the forward solve for parameter sample: {ii}")
            root_print(COMM, SEP)
            continue
        
        end = time.perf_counter() - start
            
        root_print(COMM, f"Forward solve took {end / 60:.2f} minutes.")
        root_print(COMM, SEP)
        
        root_print(COMM, f"Writing out the state...")
    
        # grab the states at observation times to write to file.
        outdata = [x0[hp.STATE].view(date) for date in visit_days]
        write_mv_to_h5(COMM, outdata, Vh[hp.STATE], STATE_DATA, name="state")
        
        if WRITE_VIZ:
            pde.exportState(x0[hp.STATE], STATE_VIZ)
            root_print(COMM, f"State (visualization) written to:\t{STATE_VIZ}")
            write_mv_to_xdmf(COMM, outdata, Vh[hp.STATE], STATE_VIZ, name="state")
        
        root_print(COMM, f"State (data) written to:\t{STATE_DATA}")
        root_print(COMM, SEP)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BIP for the synthetic experiment.")
    
    # data inputs.
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--pdir", type=str, required=True, help="Directory to where the (synthetic) patient data is stored.")
    parser.add_argument("--imgfreq", type=int, required=False, default=1, help="Frequency of images [days].")
    
    # modeling inputs.
    parser.add_argument("--experiment_type", type=str, required=True, choices=["rd", "rdtx"], help="Type of experiment to run.")
    
    # propagation options.
    parser.add_argument("--samples", type=str, required=True, help="Full path to the samples data file.")
    parser.add_argument("--nsamples", type=int, default=-1, required=True, help="Number of samples to use.")
    parser.add_argument("--sample_type", type=str, required=True, choices=["prior", "la_post", "mcmc", "map", "prior_mean"], help="Type of samples file.")
    parser.add_argument("--l2f", action=argparse.BooleanOptionalAction, default=True, help="Make prediction from last observation to holdout?")
    parser.add_argument("--pred_date", type=int, required=True, default=7, help="How many days to predict in the future?")
    
    # output options.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose output.")
    parser.add_argument("--write_viz", action=argparse.BooleanOptionalAction, default=False, help="Write visualization files?")
    
    # Parse the arguments, strip CLI args for PETSc.
    args, other = parser.parse_known_args()
    petsc4py.init(other)
    
    main(args)
