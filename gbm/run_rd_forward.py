################################################################################
# 
# This script is used to simulate the forward model for the reaction-diffusion tumor growth model.
# The script is designed to be run in parallel using MPI.
# The script will load in a mesh, set up the variational problem, and solve the forward model.
# The script will write out the initial condition, the parameter, and the state
# 
# NOTE(s)
#  -- This script currently only supports loading the initial condition file 
#    from the patient directory. That is, the initial time is always set to t0=0
# 
#  -- This script supports simulation of a single parameter sample OR
#    simulation of a continguous set of parameter samples read from file.
# 
#  -- This script currently only stores the state data at the observation times.
# 
#  -- The mesh is expected to be in HDF5 since this is intended as a post-processing script.
# 
# An example call to this script is:
# mpirun -np <num_procs> python3 run_rd_forward.py 
#           --pinfo /path/to/patient_info.json
#           --pdir /path/to/patient/data/
#           --outdir /path/to/store/results/
#           --mesh /path/to/mesh/
#           --samples /path/to/samples/file/
#           --dataname post_sample 
#           --nsamples 50
#           --uq
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
    from dt4co.utils.mesh_utils import report_mesh_info, check_mesh_dimension, load_mesh
    from dt4co.utils.data_utils import nifti2Function
    from dt4co.utils.parallel import root_print
    from dt4co.utils.fenics_io import write_mv_to_h5, read_mv_from_h5, write_mv_to_xdmf
    from dt4co.dataModel import PatientData
    
    ############################################################
    # 0. General setup.
    ############################################################
    SEP = "\n"+"#"*80+"\n"
    PREFIX = f"{args.prefix}_" if args.prefix is not None else ""
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
    DATA_NAME = args.dataname
    SAMPLE_FILE = args.samples
    SIDX = args.sidx
    NSAMPLES = args.nsamples
    UQ_MODE = args.uq
    WRITE_VIZ = args.viz
    
    os.makedirs(OUT_DIR, exist_ok=True)  # output directory
    
    # Load in the patient data.
    pinfo = PatientData(args.pinfo, args.pdir)
    
    # Experiment setup.
    factory = ExperimentFactory(pinfo)
    exp = factory.get_experiment(args.experiment_type)
    root_print(COMM, f"Using experiment: {args.experiment_type}")
    root_print(COMM, f"Experiment instance: {type(exp)}")
    
    t0 = 0.0                                # initial time
    tf = None if args.tf < 0 else args.tf   # final time
    tf = pinfo.visit_days[-1] if tf is None else tf  # grab the last visit day if nothing was specified
    
    root_print(COMM, SEP)
    
    ############################################################
    # 1. Load mesh and define function spaces.
    ############################################################
    root_print(COMM, f"Loading in the mesh...")
    root_print(COMM, SEP)
        
    mesh = load_mesh(COMM, MESH_FPATH)
    
    root_print(COMM, f"Successfully loaded the mesh.")
    root_print(COMM, f"There are {nproc} process(es).")
    
    # Get the physical dimension of the mesh.
    ZOFF = check_mesh_dimension(mesh, args.zoff)
    
    report_mesh_info(mesh)
    
    #  Set up variational spaces for state and parameter.
    Vh = exp.setupFunctionSpaces(mesh)
    
    ############################################################
    # 2. Set up the variational problem.
    ############################################################
    root_print(COMM, f"Setting up variational problem...")
    root_print(COMM, SEP)
    
    # Load in the initial condition.
    u0 = dl.Function(Vh[hp.STATE])
    nifti2Function(pinfo.get_visit(0).tumor, u0, Vh[hp.STATE], ZOFF)
    
    # Expecting solver parameters to be set from either CLI or .petscrc
    sparam = {"snes_view": None} if VERBOSE else None
    
    # Set up the variational problem and the prior.
    pde = exp.setupVariationalProblem(Vh=Vh, u0=u0, t0=t0, tf=tf, sparam=sparam, moll=args.moll)
    
    ############################################################
    # 3. Get the parameter sample.
    ############################################################
    root_print(COMM, f"Getting parameter sample...")
    root_print(COMM, SEP)
    
    # Load from file if provided, otherwise use the prior mean.
    if SAMPLE_FILE is None:
        root_print(COMM, f"No sample file provided, using the prior mean as the parameter sample.")
        mmv = hp.MultiVector(pde.generate_parameter(), 1)
        mmv[0].zero()
        mprior = exp.setupPrior(Vh)
        mmv[0].axpy(1.0, mprior.mean)
    else:
        root_print(COMM, f"Reading parameter sample from file:\t{SAMPLE_FILE}")
        if UQ_MODE:
            # forward UQ mode.
            mmv = hp.MultiVector(pde.generate_parameter(), NSAMPLES)
            mnames = [f"/data/{DATA_NAME}/{i:06d}" for i in range(SIDX, SIDX+NSAMPLES)]
            read_mv_from_h5(COMM, mmv, Vh[hp.PARAMETER], SAMPLE_FILE, name=mnames)
        else:
            # just one simulation.
            mmv = hp.MultiVector(pde.generate_parameter(), 1)
            read_mv_from_h5(COMM, mmv, Vh[hp.PARAMETER], SAMPLE_FILE, name=[DATA_NAME])
    
    ############################################################
    # 4. Solve the forward model, write out the data.
    ############################################################
    
    # loop over the multi-vector
    for ii in range(mmv.nvec()):
        m0 = mmv[ii]  # extract sample from multivector
        
        root_print(COMM, f"Beginning the forward solve for parameter sample: {SIDX + ii}")
        root_print(COMM, f"Solve {ii+1}/{mmv.nvec()} for this run.")
        root_print(COMM, SEP)
    
        # list to store solution, parameter, and adjoint.
        x0 = [pde.generate_state(), m0, None]
        
        FULL_STATE_VIZ = os.path.join(OUT_DIR, f"{PREFIX}full_state{SIDX + ii:06d}.xdmf")
        STATE_VIZ = os.path.join(OUT_DIR, f"{PREFIX}state{SIDX + ii:06d}.xdmf")
        STATE_DATA = os.path.join(OUT_DIR, f"{PREFIX}{SIDX + ii:06d}{SUFFIX}.h5")
        
        try:
            start = time.perf_counter()
            pde.solveFwd(x0[hp.STATE], x0)
            end = time.perf_counter() - start
            
            root_print(COMM, f"Forward solve took {end / 60:.2f} minutes.")
            root_print(COMM, SEP)
            
            root_print(COMM, f"Writing out the state...")    
        
            # grab the states at observation times to write to file.
            outdata = [x0[hp.STATE].view(date) for date in pinfo.visit_days]
            write_mv_to_h5(COMM, outdata, Vh[hp.STATE], STATE_DATA, name="state")
            
            if WRITE_VIZ:
                pde.exportState(x0[hp.STATE], FULL_STATE_VIZ)
                root_print(COMM, f"State (visualization) written to:\t{STATE_VIZ}")
                write_mv_to_xdmf(COMM, outdata, Vh[hp.STATE], STATE_VIZ, name="state")
            
            root_print(COMM, f"State (data) written to:\t{STATE_DATA}")
            root_print(COMM, SEP)
        except:
            root_print(COMM, f"Error in the forward solve for parameter sample: {SIDX + ii}")
            root_print(COMM, SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the forward solver.")
    
    # Directories for data.
    parser.add_argument("--pinfo", type=str, help="Path to the patient information file.")
    parser.add_argument("--pdir", type=str, help="Path to the patient data directory.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--experiment_type", type=str, required=True, help="Type of experiment to run.")
    
    parser.add_argument("--samples", type=str, default=None, help="Path to the parameter samples file.")
    parser.add_argument("--sidx", type=int, default=0, help="Index of the first sample to use.")
    parser.add_argument("--nsamples", type=int, default=1, help="How many samples to use (if running in UQ mode).")
    parser.add_argument("--uq", action=argparse.BooleanOptionalAction, default=True, help="Should the script be run in forward UQ mode (with a sampling file?).")
     
    # Output options.
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose output.")
    parser.add_argument("--prefix", type=str, default="forward", help="Name prefix for the output files.")
    parser.add_argument("--viz", action=argparse.BooleanOptionalAction, default=False, help="Write out the visualization files.")
    
    # Input options.
    parser.add_argument("--tf", type=float, default=-1.0, help="Final time for simulation [days].")
    parser.add_argument("--dataname", type=str, default="data", help="Name of the data series.")
    parser.add_argument("--zoff", type=float, default=None, help="Z-offset for 2D meshes.")
    parser.add_argument("--moll", action=argparse.BooleanOptionalAction, default=True, help="Use mollified forward model.")
    
    # Parse the arguments, strip CLI args for PETSc.
    args, other = parser.parse_known_args()
    petsc4py.init(other)
    
    main(args)
