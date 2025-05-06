################################################################################
# 
# This script is used to generate samples from the posterior using MCMC.
# 
# An example call to this script is:
# python3 draw_mcmc_samples.py \
#    --eval /path/to/eigenvalues.txt \
#    --evec /path/to/eigenvectors/ \
#    --map /path/to/map/ \
#    --mesh /path/to/mesh/ \
#    --nsamples num_samples \
#    --outdir /path/to/output/ \
#    --nmodes num_modes \
#    --pdir /path/to/patient/data \
#    --imgfreq imaging_frequence \
#    --noisy \
#    --experiment_type experiment_type 
# 
# For more information run: python3 draw_mcmc_samples.py --help
# 
################################################################################

import os
import sys
import argparse
import petsc4py         # before dolfin to avoid petsc4py.init error
import numpy as np

class QOIWrapper(object):
    """Wrapper to reject sample that breaks forward model.
    """
    def __init__(self, model):
        self.model = model
        
    def eval(self, x):
        try:
            _, _, val = self.model.cost(x)  # misfit cost
        except:
            val = 1e16
        
        return val
        

def main(args) -> None:
    import dolfin as dl

    sys.path.append(os.environ.get("HIPPYLIB_PATH"))
    import hippylib as hp

    sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
    from dt4co.utils.parallel import root_print
    from dt4co.synth import synthExperiment, setup_data_collection_timeline
    from dt4co.models import TDRealizationTracer
    from dt4co.utils.mesh_utils import report_mesh_info, load_mesh
    from dt4co.utils.data_utils import nifti2Function
    from dt4co.utils.fenics_io import write_mv_to_h5, read_mv_from_h5, write_mv_to_xdmf
    
    # -----------------------------------------------------------
    # 0. Unpack input arguments.
    # -----------------------------------------------------------
    SEP = "\n"+"#"*80+"\n"
    dl.set_log_level(dl.LogLevel.WARNING)  # suppress FEniCS output.
    
    COMM = dl.MPI.comm_world
    rank = COMM.rank
    
    MESH_FPATH = args.mesh
    
    # unpack arguments for Laplace approximation
    EVAL_FILE = args.eval
    EVEC_FILE = args.evec
    MAP_FILE = args.map
    NSAMPLES = args.nsamples
    NMODES = args.nmodes
    
    # unpack arguments for the experiment
    PATIENT_DIR = args.pdir
    EXP_TYPE = args.experiment_type
    IMG_FREQ = args.imgfreq     # imaging frequency [days] (every third day)
    NOISY_DATA = args.noisy
    
    if NOISY_DATA:
        IC_FILE = os.path.join(PATIENT_DIR, f"synthetic_noisy_obs_day_{int(0):03d}.nii")
    else:
        IC_FILE = os.path.join(PATIENT_DIR, f"synthetic_obs_day_{int(0):03d}.nii")
    
    TX_START = 14.0             # start of therapy [days]
    PRED_DATE = 14.0            # how long to predict after the last observation [days]
    N_HOLDOUT = 1               # number of holdout observations
    
    OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # set up filename convention.
    PREFIX = f"{args.prefix}_" if args.prefix is not None else ""
    SUFFIX = f"_data"  # for the HDF5 data files.
    
    # files to be generated.
    MCMC_DATA_FILE = os.path.join(OUT_DIR, f"{PREFIX}mcmc_samples{SUFFIX}.h5")
    MCMC_VIZ_FILE = os.path.join(OUT_DIR, f"{PREFIX}mcmc_samples.xdmf")
    
    # -----------------------------------------------------------
    # 1. Set up necessary experiment objects to define the model object.
    # -----------------------------------------------------------
    
    root_print(COMM, SEP)
    root_print(COMM, f"Loading in the mesh...")
    
    mesh = load_mesh(COMM, MESH_FPATH)
    
    root_print(COMM, f"Successfully loaded the mesh.")
    report_mesh_info(mesh)
    
    root_print(COMM, f"Setting up experiment and function spaces.")
    exp = synthExperiment()
    
    #  Set up variational spaces for state and parameter.
    Vh = exp.setupBIPFunctionSpaces(mesh, mle=False)
    mprior = exp.setupPrior(Vh)
    mfun = dl.Function(Vh[hp.PARAMETER])
    
    root_print(COMM, "Setting up the forward model.")
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

    tf = img_days[-1 - N_HOLDOUT]
    
    # get the initial condition.
    u0 = dl.Function(Vh[hp.STATE])
    nifti2Function(IC_FILE, u0, Vh[hp.STATE])

    pde = exp.setupBIPVariationalProblem(Vh, u0, t0, tf, exptype=EXP_TYPE, sparams=None, radio_model=stupp_radio, chemo_model=stupp_chemo)
    
    root_print(COMM, "Setting up the misfit object.")
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
    NOISE_VAR = exp.NOISE*exp.NOISE  # variance
    misfits = exp.spoofMisfitTD(visits=visits, visit_days=visit_days, Vh=Vh[hp.STATE], noise_var=NOISE_VAR, exnii=IC_FILE)
    
    # Set up the inverse problem.
    model = hp.Model(pde, mprior, misfits)
    
    # -----------------------------------------------------------
    # 2. Read back the eigenpairs, set up the Laplace approximation.
    # -----------------------------------------------------------
    
    root_print(COMM, SEP)
    root_print(COMM, f"Reading in the eigenpairs...")
    root_print(COMM, f"Eigenvalue file:\t{EVAL_FILE}")
    root_print(COMM, f"Eigenvector file:\t{EVEC_FILE}")
    
    # eigenvalues.
    evals = np.loadtxt(EVAL_FILE)
    evals = evals[:NMODES] if NMODES > 0 else evals  # subset if necessary
    num_evec = len(evals)
    
    # set up mulitvector to store the eigenvectors.
    evec = hp.MultiVector(mfun.vector(), num_evec)
    read_mv_from_h5(COMM, evec, Vh[hp.PARAMETER], EVEC_FILE, name="gen_evec")
    
    # read back the MAP point.
    mmap = hp.MultiVector(mfun.vector(), 1)
    read_mv_from_h5(COMM, mmap, Vh[hp.PARAMETER], MAP_FILE, name=["map"])
    
    nu_la = hp.GaussianLRPosterior(mprior, evals, evec)
    nu_la.mean = mmap[0]
    
    # -----------------------------------------------------------
    # 3. Draw samples from the prior & posterior, write to file.
    # -----------------------------------------------------------
    
    root_print(COMM, f"Generating samples from the Posterior with MCMC...")
    
    kgpCN_kernel = hp.gpCNKernel(model, nu_la)
    chain = hp.MCMC(kgpCN_kernel)
    chain.parameters["burn_in"] = 0
    chain.parameters["number_of_samples"] = NSAMPLES
    chain.parameters["print_progress"] = 10            
    
    dummy_state = dl.Function(Vh[hp.STATE])
    par_mv = hp.MultiVector(pde.generate_parameter(), chain.parameters["number_of_samples"])
    state_mv = hp.MultiVector(dummy_state.vector(), chain.parameters["number_of_samples"])
    
    tracer = TDRealizationTracer(chain.parameters["number_of_samples"], tf, par_mv, state_mv)
    
    if rank != 0:
        chain.parameters["print_level"] = -1
    
    QOI = QOIWrapper(model)
    
    n_accept = chain.run(mmap[0], QOI, tracer)
    
    iact, lags, acoors = hp.integratedAutocorrelationTime(tracer.data[:,0])
    
    root_print(COMM, f"Number accepted = {n_accept}")
    root_print(COMM, f"Integrated autocorrelation time = {iact}")
    
    # todo: only keep accepted samples?
    
    root_print(COMM, f"Writing samples to file...")
    root_print(COMM, f"Parameter data file:\t{MCMC_DATA_FILE}")
    write_mv_to_h5(COMM, tracer.par_mv, Vh[hp.PARAMETER], MCMC_DATA_FILE, name="mcmc_sample")
    
    # optionally, write to XDMF files for visualization.
    if args.write_viz:
        root_print(COMM, f"Writing samples to XDMF files for visualization...")
        root_print(COMM, f"Parmaeter visualization file:\t{MCMC_VIZ_FILE}")
        write_mv_to_xdmf(COMM, tracer.par_mv, Vh[hp.PARAMETER], MCMC_VIZ_FILE, name="mcmc_sample")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate samples from the posterior with gpCN MCMC.")
    
    # data inputs.
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--eval", type=str, required=True, help="File containing the eigenvalues.")
    parser.add_argument("--evec", type=str, required=True, help="File containing the eigenvectors.")
    parser.add_argument("--map", type=str, required=True, help="File containing the MAP point.")
    
    # modeling inputs.
    parser.add_argument("--nsamples", required=True, type=int, help="Number of samples to draw.")
    parser.add_argument("--nmodes", type=int, default=-1, help="Number of modes to use.")
    
    # output options.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--prefix", type=str, required=False, default="bip", help="Name prefix for the output files.")
    parser.add_argument("--write_viz", action=argparse.BooleanOptionalAction, default=False, help="Verbose output.")
    
    # required to set up the hIPPYlib model (cost function).
    parser.add_argument("--pdir", type=str, required=True, help="Directory to where the (synthetic) patient data is stored.")
    parser.add_argument("--imgfreq", type=int, required=False, default=1, help="Frequency of images [days].")
    parser.add_argument("--noisy", action=argparse.BooleanOptionalAction, default=True, help="Whether or not to use measurements polluted with noise.")
    parser.add_argument("--experiment_type", type=str, required=True, choices=["rd", "rdtx"], help="Type of experiment to run.")
    
    # Parse the arguments, strip CLI args for PETSc.
    args, other = parser.parse_known_args()
    petsc4py.init(other)
    
    main(args)
    