################################################################################
# 
# This script is used to run a finite-difference check to verify the model
# gradient and Hessian. The script is designed to be run in parallel using MPI.
# 
# An example call to this script is:
# mpirun -np <num_procs> python3 run_verify_model.py 
#           --pinfo /path/to/patient_info.json
#           --datadir /path/to/patient/data/
#           --outdir /path/to/store/results/
#           --mesh /path/to/mesh/
# 
################################################################################

import os
import sys
import time
import argparse
import petsc4py         # before dolfin to avoid petsc4py.init error
import numpy as np
import matplotlib.pyplot as plt

def main(args)->None:
    # Load these modules here so that the petsc4py.init() call can handle the CLI args.
    import dolfin as dl
    
    sys.path.append(os.environ.get("HIPPYLIB_PATH"))
    import hippylib as hp
    
    sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
    from dt4co.experiments import rdExperiment, rdtxExperiment
    from dt4co.utils.mesh_utils import report_mesh_info, check_mesh_dimension, load_mesh
    from dt4co.utils.data_utils import makeMisfitTD, nifti2Function
    from dt4co.utils.parallel import root_print
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
    
    # Paths for data.
    MESH_FPATH = args.mesh
    OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)  # output directory
    
    # Load in the patient data.
    pinfo = PatientData(args.pinfo, args.datadir)
    
    # Experiment setup.
    exp = rdExperiment()
    exp = rdtxExperiment(pinfo)
    
    t0 = 0.0                # initial time
    N_HOLDOUT = len(pinfo.visit_days) - 3  # number of holdout observations
    USE_OBSOP = args.pointwise  # whether to use pointwise observation operator.
    
    # MPI setup.
    COMM = dl.MPI.comm_world
    rank = COMM.rank
    nproc = COMM.size
    
    ############################################################
    # 1. Load mesh and define function spaces.
    ############################################################
    root_print(COMM, SEP)
    root_print(COMM, f"Loading in the mesh...")
    root_print(COMM, SEP)
    
    mesh = load_mesh(COMM, MESH_FPATH)
    
    root_print(COMM, f"Successfully loaded the mesh.")
    root_print(COMM, f"There are {nproc} process(es).")
    
    root_print(COMM, f"Writing HDF5 version of mesh to output directory:\t{OUT_DIR}")
    with dl.HDF5File(COMM, f"{OUT_DIR}/mesh.h5", "w") as fid:
        fid.write(mesh, "/mesh")
    
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
    
    # Load in data, get simulation times.
    root_print(COMM, f"Using pointwise observation operator: {USE_OBSOP}")
    misfits = makeMisfitTD(pinfo, Vh[hp.STATE], exp.bc, exp.NOISE, zoff=ZOFF, nholdout=N_HOLDOUT, pointwise=USE_OBSOP)
    tf = pinfo.visit_days[-1-N_HOLDOUT]  # final time.
    
    # Expecting solver parameters to be set from either CLI or .petscrc
    sparam = {"snes_view": None} if VERBOSE else None
    
    # Set up the variational problem and the prior.
    pde = exp.setupVariationalProblem(Vh=Vh, u0=u0, t0=t0, tf=tf, sparam=sparam, moll=args.moll)
    mprior = exp.setupPrior(Vh)
    
    ############################################################
    # 3. Set up the inverse problem and perform a FD check.
    ############################################################ 
    model = hp.Model(pde, mprior, misfits)
    
    VERIFY_FIG = os.path.join(OUT_DIR, f"{PREFIX}modelVerify.png")
    
    root_print(COMM, f"Testing the gradient and the Hessian of the model...")
    root_print(COMM, SEP)
    
    # set linearization point to the prior mean.
    m0 = dl.Function(Vh[hp.PARAMETER])
    m0.vector().zero()
    m0.vector().axpy(1., mprior.mean)
    
    eps_list = np.logspace(-10, -1, num=18)

    start = time.perf_counter()
    eps, err_grad, err_H = hp.modelVerify(model, m0.vector(), is_quadratic=False, misfit_only=True, verbose=True, eps=eps_list)
    end = time.perf_counter()
    
    root_print(COMM, f"FD check took {(end - start) / 60:.2f} minutes.")
    root_print(COMM, SEP)
    
    plt.savefig(VERIFY_FIG)
    plt.close()
            
    root_print(COMM, SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a finite-difference check to verify the model.")
    
    # Required inputs.
    parser.add_argument("--pinfo", type=str, help="Path to the patient information file.")
    parser.add_argument("--datadir", type=str, help="Path to the patient data directory.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    
    # Output options.
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose output.")
    parser.add_argument("--prefix", type=str, default="bip", help="Name prefix for the output files.")
    
    # Optional arguments.
    parser.add_argument("--zoff", type=float, default=None, help="Z-offset for 2D meshes.")
    parser.add_argument("--moll", action=argparse.BooleanOptionalAction, default=False, help="Use mollified forward model.")
    parser.add_argument("--pointwise", action=argparse.BooleanOptionalAction, default=True, help="Use pointwise observation operator.")
    parser.add_argument("--wtx", action=argparse.BooleanOptionalAction, default=False, help="Include the radio/chemo therapy models.")
    
    # Parse the arguments, strip CLI args for PETSc.
    args, other = parser.parse_known_args()
    petsc4py.init(other)
    
    main(args)
