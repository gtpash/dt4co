################################################################################
# 
# This script solves for the maximum a posteriori (MAP) point and generates
# the Laplace approximation of the posterior for the synthetic experiment.
# 
# NOTE: This script uses a continuous observation operator and first solves the
#       forward problem to generate synthetic observations.
# 
# An example call to this script is:
# mpirun -np <num_procs> python3 run_synth_bip.py
#       --pdir /path/to/patient/data/
#       --outdir /path/to/store/results/
#       --synthdatadir /path/to/synthetic/data/
#       --imgfreq IMAGING_FREQUENCY
#       --experiment_type EXPERIMENT_TYPE
# 
# For more information run: python3 run_cont_synth_bip.py --help
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
    from mpi4py import MPI
    
    sys.path.append(os.environ.get("HIPPYLIB_PATH"))
    import hippylib as hp
    
    sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
    from dt4co.synth import synthExperiment, setup_data_collection_timeline
    from dt4co.utils.mesh_utils import report_mesh_info
    from dt4co.utils.fenics_io import write_mv_to_h5, write_mv_to_xdmf
    from dt4co.utils.parallel import root_print
    from dt4co.utils.model_utils import MollifierInitialCondition
    
    # ------------------------------------------------------------
    # General setup.
    # ------------------------------------------------------------
    SEP = "\n"+"#"*80+"\n"
    
    PREFIX = f"{args.prefix}_" if args.prefix is not None else ""
    SUFFIX = f"_data"  # for the HDF5 data files.
    
    dl.set_log_level(dl.LogLevel.WARNING)  # suppress FEniCS output.
    VERBOSE = args.verbose
    
    # MPI setup.
    COMM = dl.MPI.comm_world
    rank = COMM.rank
    nproc = COMM.size
    
    root_print(COMM, SEP)
    
    # Paths for data.
    OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)  # output directory

    EXP_TYPE = args.experiment_type

    # ------------------------------------------------------------
    # Set up the experiment.
    # ------------------------------------------------------------
    
    NX = 100
    NY = 100
    L = 200
    
    mesh = dl.RectangleMesh(dl.Point(0., 0.), dl.Point(L, L), NX, NY)
    report_mesh_info(mesh)
    
    IMG_FREQ = args.imgfreq     # imaging frequency [days] (every third day)
    TX_START = 14.0  # start of therapy [days]
    PRED_DATE = 14.0    # how long to predict after the last observation [days]
    
    exp = synthExperiment()
    
    NSAMPLES = 300  # number of samples to draw from the prior and posterior.
    
    ###########################################################################
    # ------------------------------------------------------------
    # Generate data from the true model
    # ------------------------------------------------------------
    ###########################################################################
    
    root_print(COMM, "Generating synthetic data from the true model.")
    Vh = exp.setupBIPFunctionSpaces(mesh)

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
    
    # Place a seed tumor in the middle of the domain.
    ctr = MPI.COMM_WORLD.allreduce( np.mean(mesh.coordinates(), axis=0), op=MPI.SUM ) / nproc # roughly the center of the mesh.
    
    u0_expr = MollifierInitialCondition(dim=2, center=ctr, r=L/16, v=20, degree=2)
    u0 = dl.interpolate(u0_expr, Vh[hp.STATE])

    # set up the ground truth model
    sparam = {"snes_view": None} if VERBOSE else None
    pde = exp.setupBIPVariationalProblem(Vh, u0, t0, tf, exptype=EXP_TYPE, radio_model=stupp_radio, chemo_model=stupp_chemo, sparams=sparam)

    mtrue = dl.Function(Vh[hp.PARAMETER])    
    mtrue.assign(dl.Constant([np.log(exp.DG_TRUE), np.log(exp.K_TRUE)]))
    
    utrue = pde.generate_state()
    x0 = [utrue, mtrue.vector(), None]

    root_print(COMM, "Beginning forward solve.")
    start = time.perf_counter()
    pde.solveFwd(x0[hp.STATE], x0)
    end = time.perf_counter() - start
    root_print(COMM, f"Forward solve took {end / 60:.2f} minutes.")
    root_print(COMM, SEP)
    
    pde.exportState(x0[hp.STATE], os.path.join(OUT_DIR, "synthetic_state.xdmf"))
    
    # ------------------------------------------------------------
    # Set up the function spaces.
    # ------------------------------------------------------------
    root_print(COMM, "Setting up function spaces and tissue segmentation indicator function.")
    Vh = exp.setupBIPFunctionSpaces(mesh, args.mle)
    
    N_HOLDOUT = 1                         # number of holdout observations
    tf = img_days[-1 - N_HOLDOUT]
    
    # Inverse problem parameters.
    bip_parameters = hp.ReducedSpaceNewtonCG_ParameterList()
    # bip_parameters["rel_tolerance"] = 1e-9
    # bip_parameters["abs_tolerance"] = 1e-12
    bip_parameters["max_iter"]      = 40
    bip_parameters["cg_max_iter"]   = 75
    bip_parameters["globalization"] = "LS"
    bip_parameters["LS"]["max_backtracking_iter"] = 15
    bip_parameters["GN_iter"] = 5
    if rank != 0:
        bip_parameters["print_level"] = -1
    else:
        root_print(COMM, SEP)
        root_print(COMM, "Inverse problem parameters:")
        bip_parameters.showMe()
        root_print(COMM, SEP)
        
    # Laplace approximation parameters.
    kk = 25                      # number of eigenvectors
    pp = 5                       # oversampling for randomized method.
    
    # ------------------------------------------------------------
    # Set up the variational problem.
    # ------------------------------------------------------------
    root_print(COMM, "Setting up the forward model.")

    # sparam = {"snes_view": None} if VERBOSE else None
    sparam = None
    pde = exp.setupBIPVariationalProblem(Vh, u0, t0, tf, exptype=EXP_TYPE, sparams=sparam, radio_model=stupp_radio, chemo_model=stupp_chemo)
    root_print(COMM, "Set up the variational problem.")
    
    mprior = exp.setupPrior(Vh, args.mle)
    root_print(COMM, "Set up the prior.")
    
    # ------------------------------------------------------------
    # Spoof the misfits object.
    # ------------------------------------------------------------
    root_print(COMM, "Setting up the misfit object.")
    
    # set up the visit days for the misfit
    visit_days = img_days[1:-N_HOLDOUT]  # skip the first day, hold outs for misfit
    
    # Print out the visit dates
    for i, date in enumerate(visit_days):
        root_print(COMM, f"Visit {i+1}: {date}")

    root_print(COMM, "Setting up the misfit object.")
    
    max_state = x0[hp.STATE].norm("linf", "linf")
    noise_std_dev = exp.NOISE * max_state
    NOISE_VAR = noise_std_dev  # variance
    
    noisy_utrue = x0[hp.STATE].copy()
    if args.noisy:
        hp.parRandom.normal_perturb(exp.NOISE, noisy_utrue)
        
    pde.exportState(noisy_utrue, os.path.join(OUT_DIR, "noisy_state.xdmf"))
    
    misfits = exp.spoofContinuousMisfitTD(uh=noisy_utrue, visit_days=visit_days, Vh=Vh[hp.STATE], noise_var=NOISE_VAR)
    
    # ------------------------------------------------------------
    # Solve the deterministic inverse problem.
    # ------------------------------------------------------------
    root_print(COMM, f"Finding the MAP point...")
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
    
    # ------------------------------------------------------------
    # Write the prior mean, MAP point, and state to file.
    # ------------------------------------------------------------
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
        # ------------------------------------------------------------
        # Compute the low rank Gaussian Approximation of the posterior.
        # ------------------------------------------------------------
        root_print(COMM, f"Computing the Laplace approximation of the posterior...")

        start = time.perf_counter()

        # Set the linearization point.
        model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
        Hmisfit = hp.ReducedHessian(model, misfit_only=True)
        
        # Run the double pass algorithm to generate the eigenpairs.
        root_print(COMM, f"Double Pass Algorithm. Requested eigenvectors: {kk}, oversampling {pp}.")
        
        # m_dummy = dl.Function(Vh[hp.PARAMETER])
        # Omega = hp.MultiVector(m_dummy.vector(), kk+pp)
        # Omega.zero()
        # hp.parRandom.normal(1., Omega)
        
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
        
        # ------------------------------------------------------------
        # Save the eigenvectors.
        # ------------------------------------------------------------
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
        
        # -----------------------------------------------------------
        # Draw samples from the prior & posterior, write to file.
        # -----------------------------------------------------------
        
        root_print(COMM, f"Generating samples from Prior and Posterior...")
        
        # files to be generated.
        PRIOR_DATA_FILE = os.path.join(OUT_DIR, f"{PREFIX}prior_samples{SUFFIX}.h5")
        POST_DATA_FILE = os.path.join(OUT_DIR, f"{PREFIX}posterior_samples{SUFFIX}.h5")

        # set up helpers.
        noise = dl.Vector()
        posterior.init_vector(noise, "noise")   
        s_prior_fun = dl.Function(Vh[hp.PARAMETER])
        s_post_fun = dl.Function(Vh[hp.PARAMETER])
        
        # set up MultiVectors to store the samples.
        s_prior = hp.MultiVector(s_prior_fun.vector(), NSAMPLES)
        s_post = hp.MultiVector(s_post_fun.vector(), NSAMPLES)
        
        for i in range(NSAMPLES):
            if i % 10 == 0:
                root_print(COMM, f"Generating sample {i+1}/{NSAMPLES}")
            hp.parRandom.normal(1., noise)
            posterior.sample(noise, s_prior[i], s_post[i])
        
        root_print(COMM, f"Writing samples to file...")
        root_print(COMM, f"Prior data file:\t{PRIOR_DATA_FILE}")
        root_print(COMM, f"Posterior data file:\t{POST_DATA_FILE}")
        write_mv_to_h5(COMM, s_prior, Vh[hp.PARAMETER], PRIOR_DATA_FILE, name="prior_sample")
        write_mv_to_h5(COMM, s_post, Vh[hp.PARAMETER], POST_DATA_FILE, name="post_sample")
        
        # if verbose, write to XDMF files for visualization.
        if args.verbose:
            PRIOR_VIZ_FILE = os.path.join(OUT_DIR, f"{PREFIX}prior_samples.xdmf")
            POST_VIZ_FILE = os.path.join(OUT_DIR, f"{PREFIX}posterior_samples.xdmf")
            root_print(COMM, f"Writing samples to XDMF files for visualization...")
            root_print(COMM, f"Prior visualization file:\t{PRIOR_VIZ_FILE}")
            root_print(COMM, f"Posterior visualization file:\t{POST_VIZ_FILE}")
            
            write_mv_to_xdmf(COMM, s_prior, Vh[hp.PARAMETER], PRIOR_VIZ_FILE, name="prior_sample")
            write_mv_to_xdmf(COMM, s_post, Vh[hp.PARAMETER], POST_VIZ_FILE, name="post_sample")
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BIP for the synthetic experiment.")
    
    # Required inputs.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--experiment_type", type=str, required=True, choices=["rd", "rdtx"], help="Type of experiment to run.")
    parser.add_argument("--imgfreq", type=int, default=1, help="Frequency of images [days].")
    
    # Output options.
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose output.")
    parser.add_argument("--prefix", type=str, default="bip", help="Name prefix for the output files.")
    
    # Input options.
    parser.add_argument("--mle", action=argparse.BooleanOptionalAction, default=False, help="Run in MLE mode (homogeneous parameters).")
    parser.add_argument("--moll", action=argparse.BooleanOptionalAction, default=False, help="Use mollified forward model.")
    parser.add_argument("--noisy", action=argparse.BooleanOptionalAction, default=True, help="Whether or not to pollute with noise.")
    
    # Parse the arguments, strip CLI args for PETSc.
    args, other = parser.parse_known_args()
    petsc4py.init(other)
    
    main(args)
