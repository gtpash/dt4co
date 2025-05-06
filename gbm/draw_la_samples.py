################################################################################
# 
# This script is used to generate samples from the prior and Laplace approximation of the posterior.
# 
# An example call to this script is:
# python3 draw_la_samples.py \
#    --eval /path/to/eigenvalues.txt \
#    --evec /path/to/eigenvectors/ \
#    --map /path/to/map/ \
#    --mesh /path/to/mesh/ \
#    --nsamples num_samples \
#    --outdir /path/to/output/ \
#    --nmodes num_modes
# 
# For more information run: python3 draw_la_samples.py --help
# 
################################################################################

import os
import sys
import argparse

import dolfin as dl
import numpy as np

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
import hippylib as hp

sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
from dt4co.dataModel import PatientData
from dt4co.experiments import ExperimentFactory
from dt4co.utils.fenics_io import write_mv_to_h5, read_mv_from_h5, write_mv_to_xdmf
from dt4co.utils.parallel import root_print


def main(args) -> None:
    
    # -----------------------------------------------------------
    # 0. Setup.
    # -----------------------------------------------------------
    SEP = "\n"+"#"*80+"\n"
    
    COMM = dl.MPI.comm_world
    
    # unpack arguments
    EVAL_FILE = args.eval
    EVEC_FILE = args.evec
    MAP_FILE = args.map
    NSAMPLES = args.nsamples
    MESH_FPATH = args.mesh
    NMODES = args.nmodes
    
    OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # set up filename convention.
    PREFIX = f"{args.prefix}_" if args.prefix is not None else ""
    SUFFIX = f"_data"  # for the HDF5 data files.
    
    # files to be generated.
    PRIOR_DATA_FILE = os.path.join(OUT_DIR, f"{PREFIX}prior_samples{SUFFIX}.h5")
    POST_DATA_FILE = os.path.join(OUT_DIR, f"{PREFIX}la_post_samples{SUFFIX}.h5")
    PRIOR_VIZ_FILE = os.path.join(OUT_DIR, f"{PREFIX}prior_samples.xdmf")
    POST_VIZ_FILE = os.path.join(OUT_DIR, f"{PREFIX}la_post_samples.xdmf")
    
    # -----------------------------------------------------------
    # 1. Set up necessary experiment objects.
    # -----------------------------------------------------------
    
    pinfo = PatientData(args.pinfo, args.pdir)
    
    # Experiment setup.
    factory = ExperimentFactory(pinfo)
    exp = factory.get_experiment(args.experiment_type)
    root_print(COMM, f"Using experiment: {args.experiment_type}")
    root_print(COMM, f"Experiment instance: {type(exp)}")
    
    #  Set up variational spaces for state and parameter.
    root_print(COMM, SEP)
    root_print(COMM, f"Loading in the mesh...")
    mesh, _ = exp.setupMesh(COMM, MESH_FPATH, None)
    root_print(COMM, f"Setting up Prior...")
    Vh = exp.setupFunctionSpaces(mesh)
    mprior = exp.setupPrior(Vh)
    mfun = dl.Function(Vh[hp.PARAMETER])
    
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
    
    root_print(COMM, f"Generating samples from Prior and Laplace appoximation to the Posterior...")

    # set up helpers.
    noise = dl.Vector()
    nu_la.init_vector(noise, "noise")   
    s_prior_fun = dl.Function(Vh[hp.PARAMETER])
    s_post_fun = dl.Function(Vh[hp.PARAMETER])
    
    # set up MultiVectors to store the samples.
    s_prior = hp.MultiVector(s_prior_fun.vector(), NSAMPLES)
    s_post = hp.MultiVector(s_post_fun.vector(), NSAMPLES)
    
    for i in range(NSAMPLES):
        if i % 10 == 0:
            root_print(COMM, f"Generating sample {i+1}/{NSAMPLES}")
        hp.parRandom.normal(1., noise)
        nu_la.sample(noise, s_prior[i], s_post[i], add_mean=True)
    
    root_print(COMM, f"Writing samples to file...")
    root_print(COMM, f"Prior data file:\t{PRIOR_DATA_FILE}")
    root_print(COMM, f"Posterior data file:\t{POST_DATA_FILE}")
    write_mv_to_h5(COMM, s_prior, Vh[hp.PARAMETER], PRIOR_DATA_FILE, name="prior_sample")
    write_mv_to_h5(COMM, s_post, Vh[hp.PARAMETER], POST_DATA_FILE, name="la_post_sample")
    
    # optionally, write to XDMF files for visualization.
    if args.write_viz:
        root_print(COMM, f"Writing samples to XDMF files for visualization...")
        root_print(COMM, f"Prior visualization file:\t{PRIOR_VIZ_FILE}")
        root_print(COMM, f"Posterior visualization file:\t{POST_VIZ_FILE}")
        
        write_mv_to_xdmf(COMM, s_prior, Vh[hp.PARAMETER], PRIOR_VIZ_FILE, name="prior_sample")
        write_mv_to_xdmf(COMM, s_post, Vh[hp.PARAMETER], POST_VIZ_FILE, name="la_post_sample")
            
    root_print(COMM, SEP)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate samples from the prior and Laplace approximation of the posterior.")
    
    # data inputs.
    parser.add_argument("--pinfo", type=str, help="Path to the patient information file.")
    parser.add_argument("--pdir", type=str, help="Path to the patient data directory.")
    parser.add_argument("--eval", type=str, help="File containing the eigenvalues.")
    parser.add_argument("--evec", type=str, help="File containing the eigenvectors.")
    parser.add_argument("--map", type=str, help="File containing the MAP point.")
    parser.add_argument("--mesh", type=str, help="Path to the mesh file.")
    
    # modeling inputs.
    parser.add_argument("--experiment_type", type=str, required=True, choices=["rd", "rdtx", "pwrdtx"], help="Type of experiment to run.")
    parser.add_argument("--nsamples", type=int, help="Number of samples to draw.")
    parser.add_argument("--nmodes", type=int, default=-1, help="Number of modes to use.")
    
    # output options.
    parser.add_argument("--outdir", type=str, help="Output directory.")
    parser.add_argument("--prefix", type=str, default="bip", help="Name prefix for the output files.")
    parser.add_argument("--write_viz", action=argparse.BooleanOptionalAction, default=False, help="Write visualization XDMF?")
    
    args = parser.parse_args()
    main(args)
    