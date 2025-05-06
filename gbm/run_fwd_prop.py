################################################################################
# 
# This script is used to simulate the forward model for the reaction-diffusion tumor growth model.
# In particular, this script is intended for forward propagation of uncertainty through the model.
# The script is designed to be run in parallel using MPI.
# The script will load in a mesh, set up the variational problem, and solve the forward model.
# 
# An example call to this script is:
# mpirun -np <num_procs> python3 run_fwd_prop.py
#        --mesh /path/to/mesh/
#        --pinfo /path/to/patient_info.json
#        --pdir /path/to/patient/data/
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
################################################################################

import os
import sys
import time
import argparse

import petsc4py     # before dolfin to avoid petsc4py.init error
import dolfin as dl 

def main(args)->None:
    # Load these modules here so that the petsc4py.init() call can handle the CLI args.
    sys.path.append(os.environ.get("HIPPYLIB_PATH"))
    import hippylib as hp
    
    sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
    from dt4co.experiments import ExperimentFactory
    from dt4co.dataModel import PatientData
    from dt4co.utils.data_utils import nifti2Function
    from dt4co.utils.parallel import root_print
    from dt4co.utils.fenics_io import write_mv_to_h5, read_mv_from_h5, write_mv_to_xdmf, getGroupSize
    
    ############################################################
    # 0. General setup.
    ############################################################
    SEP = "\n"+"#"*80+"\n"
    SUFFIX = f"_data"  # for the HDF5 data files.
    
    # Logging.
    dl.set_log_level(dl.LogLevel.WARNING)
    VERBOSE = args.verbose
    
    # MPI setup.
    COMM = dl.MPI.comm_world
    rank = COMM.rank
    nproc = COMM.size
    
    # Paths for data.
    MESH_FPATH = args.mesh
    OUT_DIR = args.outdir
    
    IS_L2F = args.l2f
    SAMPLE_TYPE = args.sample_type
    NSAMPLES = args.nsamples
    WRITE_VIZ = args.write_viz
    
    N_HOLDOUT = 1  # number of holdout samples.
        
    os.makedirs(OUT_DIR, exist_ok=True)  # output directory
    
    # Load in the patient data.
    pinfo = PatientData(args.pinfo, args.pdir)
    
    # Experiment setup.
    factory = ExperimentFactory(pinfo)
    exp = factory.get_experiment(args.experiment_type)
    root_print(COMM, f"Using experiment: {args.experiment_type}")
    root_print(COMM, f"Experiment instance: {type(exp)}")
    
    VIDX0 = -1 - N_HOLDOUT if IS_L2F else 0
    t0 = pinfo.visit_days[VIDX0]    # initial time
    tf = pinfo.visit_days[-1]       # final time
    
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
    STATE_DIR = os.path.join(OUT_DIR, f"last_to_final") if IS_L2F else os.path.join(OUT_DIR, "full")
    PROP_DIR = os.path.join(STATE_DIR, SAMPLE_TYPE)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(PROP_DIR, exist_ok=True)
    
    root_print(COMM, SEP)
    
    # -----------------------------------------------------------
    # Load mesh and define function spaces.
    # -----------------------------------------------------------
    root_print(COMM, f"Loading in the mesh...")
    root_print(COMM, SEP)
    
    mesh, ZOFF = exp.setupMesh(COMM, MESH_FPATH, zoff=args.zoff)
    
    #  Set up variational spaces for state and parameter.
    Vh = exp.setupFunctionSpaces(mesh, mle=False)
    
    # -----------------------------------------------------------
    # Set up the variational problem.
    # -----------------------------------------------------------
    root_print(COMM, f"Setting up variational problem...")
    root_print(COMM, SEP)
    
    # Load in the initial condition.
    u0 = dl.Function(Vh[hp.STATE])
    nifti2Function(pinfo.get_visit(VIDX0).tumor, u0, Vh[hp.STATE], ZOFF)
    
    # Expecting solver parameters to be set from either CLI or .petscrc
    sparam = {"snes_view": None} if VERBOSE else None
    
    # Set up the variational problem and the prior.
    pde = exp.setupVariationalProblem(Vh=Vh, u0=u0, t0=t0, tf=tf, sparam=sparam, moll=args.moll)
    
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
    # Run the model forward, write out the data
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
        if IS_L2F:
            outdata = [ x0[hp.STATE].view(pinfo.visit_days[-1]) ]  # only need the final state.
        else:
            outdata = [x0[hp.STATE].view(date) for date in pinfo.visit_days]
            
        write_mv_to_h5(COMM, outdata, Vh[hp.STATE], STATE_DATA, name="state")
        
        if WRITE_VIZ:
            pde.exportState(x0[hp.STATE], STATE_VIZ)
            root_print(COMM, f"State (visualization) written to:\t{STATE_VIZ}")
            write_mv_to_xdmf(COMM, outdata, Vh[hp.STATE], STATE_VIZ, name="state")
        
        root_print(COMM, f"State (data) written to:\t{STATE_DATA}")
        root_print(COMM, SEP)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the forward solver.")
    
    # data inputs.
    parser.add_argument("--pinfo", type=str, help="Path to the patient information file.")
    parser.add_argument("--pdir", type=str, help="Path to the patient data directory.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    
    # modeling inputs.
    parser.add_argument("--experiment_type", type=str, required=True, choices=["rd", "rdtx", "pwrdtx"], help="Type of experiment to run.")
    parser.add_argument("--zoff", type=float, default=None, help="Z-offset for 2D meshes.")
    parser.add_argument("--moll", action=argparse.BooleanOptionalAction, default=True, help="Use mollified forward model.")
    
    # propagation options.
    parser.add_argument("--samples", type=str, required=True, help="Full path to the samples data file.")
    parser.add_argument("--nsamples", type=int, default=-1, required=True, help="Number of samples to use.")
    parser.add_argument("--sample_type", type=str, required=True, choices=["prior", "la_post", "mcmc", "map", "prior_mean"], help="Type of samples file.")
    parser.add_argument("--l2f", action=argparse.BooleanOptionalAction, default=True, help="Make prediction from last observation to holdout?")
    
    # output options.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose output.")
    parser.add_argument("--write_viz", action=argparse.BooleanOptionalAction, default=False, help="Write visualization files?")    
    
    # Parse the arguments, strip CLI args for PETSc.
    args, other = parser.parse_known_args()
    petsc4py.init(other)
    
    main(args)
