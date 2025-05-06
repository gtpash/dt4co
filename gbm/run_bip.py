################################################################################
# 
# This is the main script.
# This script is to determine the low-rank approximation to the posterior.
# The script is designed to be run in parallel using MPI.
# The script will
#   - load in a mesh
#   - set up the variational problem
#   - solve the inverse problem for the MAP point
#   - write the prior mean, map point, and state to file for post-processing
#   - compute the low-rank approximation to the posterior
# 
# Usage: mpirun -np <num_ranks> python3 run_bip.py 
#           --pinfo /path/to/patient_info.json
#           --pdir /path/to/patient/data/
#           --outdir /path/to/store/results/
#           --mesh /path/to/mesh/
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
    import numpy as np
    
    sys.path.append(os.environ.get("HIPPYLIB_PATH"))
    import hippylib as hp
    
    sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
    from dt4co.dataModel import PatientData
    from dt4co.experiments import ExperimentFactory
    from dt4co.utils.data_utils import makeMisfitTD, nifti2Function
    from dt4co.utils.parallel import root_print
    from dt4co.utils.fenics_io import write_mv_to_h5, write_mv_to_xdmf
    
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
    
    root_print(COMM, SEP)
    
    # Paths for data.
    MESH_FPATH = args.mesh
    OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)  # output directory
    
    # Load in the patient data.
    pinfo = PatientData(args.pinfo, args.pdir)
    
    # Optionally truncate radiotherapy visits.
    if args.postrt:
        pinfo.truncate_radio_visits()
    
    # Experiment setup.
    factory = ExperimentFactory(pinfo)
    exp = factory.get_experiment(args.experiment_type)
    root_print(COMM, f"Using experiment: {args.experiment_type}")
    root_print(COMM, f"Experiment instance: {type(exp)}")
        
    t0 = 0.0                    # initial time
    N_HOLDOUT = 1               # number of holdout observations
    USE_OBSOP = args.pointwise  # whether to use pointwise observation operator.

    
    # Laplace approximation parameters.
    kk = args.nmodes                      # number of eigenvectors
    pp = args.noversample                 # oversampling for randomized method.
    
    ############################################################
    # 1. Load mesh and define function spaces.
    ############################################################
    root_print(COMM, f"Loading in the mesh...")
    root_print(COMM, SEP)
    
    mesh, ZOFF = exp.setupMesh(COMM, MESH_FPATH, zoff=args.zoff)
    
    #  Set up variational spaces for state and parameter.
    Vh = exp.setupFunctionSpaces(mesh, mle=args.mle)
    
    ############################################################
    # 2. Set up the variational problem.
    ############################################################
    root_print(COMM, f"Setting up variational problem...")
    
    # Load in the initial condition.
    u0 = dl.Function(Vh[hp.STATE])
    nifti2Function(pinfo.get_visit(0).tumor, u0, Vh[hp.STATE], ZOFF)
    
    root_print(COMM, f"Making misfit...")
    root_print(COMM, f"Using pointwise observation operator: {USE_OBSOP}")
    # Load in data, get simulation times.
    misfits = makeMisfitTD(pinfo, Vh[hp.STATE], exp.bc, exp.NOISE, zoff=ZOFF, nholdout=N_HOLDOUT, pointwise=USE_OBSOP)
    tf = pinfo.visit_days[-1-N_HOLDOUT]  # final time.
    
    # Expecting solver parameters to be set from either CLI or .petscrc
    sparam = {"snes_view": None} if VERBOSE else None
    
    # Set up the variational problem and the prior.
    root_print(COMM, f"Making variational problem...")
    pde = exp.setupVariationalProblem(Vh=Vh, u0=u0, t0=t0, tf=tf, sparam=sparam, moll=args.moll)
    
    root_print(COMM, f"Setup prior...")
    mprior = exp.setupPrior(Vh, mle=args.mle)
    
    root_print(COMM, SEP)
    
    ############################################################
    # 3. Solve the Bayesian inverse problem.
    ############################################################
    root_print(COMM, f"Finding the MAP point...")
    root_print(COMM, SEP)
    
    # Inverse problem parameters.
    bip_parameters = exp.getBIPParameters()
    if rank != 0:
        bip_parameters["print_level"] = -1
    else:
        root_print(COMM, SEP)
        root_print(COMM, "Inverse problem parameters:")
        bip_parameters.showMe()
        root_print(COMM, SEP)
    
    # Set up the inverse problem.
    model = hp.Model(pde, mprior, misfits)
    
    # Set up the Newton-Krylov solver.
    solver = hp.ReducedSpaceNewtonCG(model, bip_parameters)
    
    # Solve the inverse problem.
    m = model.prior.mean.copy()
    
    start = time.perf_counter()
    x = solver.solve([None, m, None])
    end = time.perf_counter()
    
    if solver.converged:
        root_print(COMM, f"Converged in {solver.it} iterations.")
    else:
        root_print(COMM, "Not Converged")
    
    root_print(COMM, f"Termination reason: {solver.termination_reasons[solver.reason]}")
    root_print(COMM, f"Final gradient norm: {solver.final_grad_norm}")
    root_print(COMM, f"Final cost: {solver.final_cost}")
    root_print(COMM, f"\nInverse solve took {(end - start) / 60:.2f} minutes.")
    root_print(COMM, SEP)
    
    ############################################################
    # 4. Write the prior mean, MAP point, and state to file.
    ############################################################
    MAP_FILE = os.path.join(OUT_DIR, f"{PREFIX}map.xdmf")
    PARAM_FILE = os.path.join(OUT_DIR, f"{PREFIX}param.xdmf")
    MAP_STATE = os.path.join(OUT_DIR, f"{PREFIX}map_state.xdmf")
    PARAM_FILE_H5 = f"{os.path.splitext(PARAM_FILE)[0]}{SUFFIX}.h5"
    root_print(COMM, f"Writing the prior mean, MAP point, and state to file.")
    root_print(COMM, f"Parameter file:\t{PARAM_FILE}")
    root_print(COMM, f"State file:\t{MAP_STATE}")
    root_print(COMM, f"HDF5 with data:\t{PARAM_FILE_H5}")
    root_print(COMM, SEP)
    
    write_mv_to_xdmf(COMM, [x[hp.PARAMETER]], Vh[hp.PARAMETER], MAP_FILE, name="map")
    write_mv_to_xdmf(COMM, [mprior.mean], Vh[hp.PARAMETER], PARAM_FILE, name="prior_mean")
    write_mv_to_h5(COMM, [x[hp.PARAMETER], mprior.mean], Vh[hp.PARAMETER], PARAM_FILE_H5, name=["map", "prior_mean"])
        
    pde.exportState(x[hp.STATE], MAP_STATE)
    
    if not args.mle:
        ############################################################
        # 5. Compute the low rank Gaussian Approximation of the posterior.
        ############################################################
        root_print(COMM, f"Computing the Laplace approximation of the posterior...")

        start = time.perf_counter()

        # Set the linearization point.
        model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
        Hmisfit = hp.ReducedHessian(model, misfit_only=True)
        
        # Run the double pass algorithm to generate the eigenpairs.
        root_print(COMM, f"Double Pass Algorithm. Requested eigenvectors: {kk}, oversampling {pp}.")
        Omega = hp.MultiVector(x[hp.PARAMETER], kk+pp)
        hp.parRandom.normal(1., Omega)
        d, U = hp.doublePassG(Hmisfit, mprior.R, mprior.Rsolver, Omega, kk, s=1, check=False)
        posterior = hp.GaussianLRPosterior(mprior, d, U)
        posterior.mean = x[hp.PARAMETER]
        
        # Compute the traces.
        post_tr, prior_tr, corr_tr = posterior.trace(method="Randomized", r=200)
        root_print(COMM, f"Posterior trace: {post_tr}; Prior trace: {prior_tr}; Correction trace: {corr_tr}")
        
        # Compute the pointwise variance. 
        post_pw_variance, pr_pw_variance, corr_pw_variance = posterior.pointwise_variance(method="Randomized", r=200)
        
        # Report Results.
        kl_dist = posterior.klDistanceFromPrior()
        root_print(COMM, f"KL distance: {kl_dist}")
        
        end = time.perf_counter()
        
        root_print(COMM, f"Processing Laplace approximation took {(end - start) / 60:.2f} minutes.")
        
        # Write pointwise variance to file.
        PW_FILE = os.path.join(OUT_DIR, f"{PREFIX}pointwise_variance.xdmf")
        root_print(COMM, f"Writing pointwise variance to file:\t{PW_FILE}.")
        
        # Just for visualization.
        with dl.XDMFFile(COMM, PW_FILE) as fid:
            fid.parameters["functions_share_mesh"] = True
            fid.parameters["rewrite_function_mesh"] = False
            fid.write(hp.vector2Function(post_pw_variance, Vh[hp.PARAMETER], name="Posterior"), 0)
            fid.write(hp.vector2Function(pr_pw_variance, Vh[hp.PARAMETER], name="Prior"), 0)
            fid.write(hp.vector2Function(corr_pw_variance, Vh[hp.PARAMETER], name="Correction"), 0)
        
        ############################################################
        # 6. Save the eigenvectors.
        ############################################################
        EVEC_VIZ_FILE = os.path.join(OUT_DIR, f"{PREFIX}eigenvectors.xdmf")
        EVEC_DATA_FILE = os.path.join(OUT_DIR, f"{PREFIX}eigenvectors{SUFFIX}.h5")
        EVAL_FILE = os.path.join(OUT_DIR, f"{PREFIX}eigenvalues.txt")
        
        root_print(COMM, "Exporting generalized Eigenpairs...")
        root_print(COMM, f"Eigenvalue file:\t{EVAL_FILE}")
        root_print(COMM, f"Eigenvector visualization file:\t{EVEC_VIZ_FILE}")
        root_print(COMM, f"Eigenvector data file:\t{EVEC_DATA_FILE}")
        
        write_mv_to_h5(COMM, U, Vh[hp.PARAMETER], EVEC_DATA_FILE, name="gen_evec")
        
        U.export(Vh[hp.PARAMETER], EVEC_VIZ_FILE, varname="gen_evec", normalize=True)
        if rank == 0:
            np.savetxt(EVAL_FILE, d)
        
        root_print(COMM, SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Bayesian inverse problem for the RD tumor equation.")
    
    # data inputs.
    parser.add_argument("--pinfo", type=str, help="Path to the patient information file.")
    parser.add_argument("--pdir", type=str, help="Path to the patient data directory.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    
    # modeling inputs.
    parser.add_argument("--experiment_type", type=str, required=True, choices=["rd", "rdtx", "pwrdtx"], help="Type of experiment to run.")
    parser.add_argument("--mle", action=argparse.BooleanOptionalAction, default=False, help="Run in MLE mode (homogeneous parameters).")
    parser.add_argument("--moll", action=argparse.BooleanOptionalAction, default=False, help="Use mollified forward model.")
    parser.add_argument("--zoff", type=float, default=None, help="Z-offset for 2D meshes.")
    parser.add_argument("--pointwise", action=argparse.BooleanOptionalAction, default=True, help="Use pointwise observation operator.")
    parser.add_argument("--postrt", action=argparse.BooleanOptionalAction, default=False, help="Only use observations post-radiotherapy.")
    
    # output options.
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose output.")
    parser.add_argument("--prefix", type=str, default="bip", help="Name prefix for the output files.")
    parser.add_argument("--nmodes", type=int, default=50, help="Number of modes to use.")
    parser.add_argument("--noversample", type=int, default=10, help="Number of oversamples for randomized method.")
    
    # Parse the arguments, strip CLI args for PETSc.
    args, other = parser.parse_known_args()
    petsc4py.init(other)
    
    main(args)
