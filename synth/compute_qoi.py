################################################################################
# 
# todo; update to work in non-l2f mode
# 
# This script is intended to compute the quanities of interest at a single visit
# given the state data from the forward propagation.
# 
# An example call to this script is:
# mpirun -np <num_procs> python3 compute_qoi.py
#        --mesh /path/to/mesh/
#        --datafile /path/to/datafile/
#        --refnii /path/to/reference.nii
#        --roinii /path/to/roi.nii
#        --outdir /path/to/store/results/
# 
# NOTE: Rasterization must be done in serial.
# 
################################################################################

import os
import sys
import time
import argparse
import numpy as np

def main(args)->None:
    # Load these modules here so that the petsc4py.init() call can handle the CLI args.
    import dolfin as dl
    
    sys.path.append(os.environ.get("HIPPYLIB_PATH"))
    import hippylib as hp
    
    sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
    from dt4co.synth import synthExperiment
    from dt4co.utils.mesh_utils import report_mesh_info, load_mesh
    from dt4co.utils.data_utils import nifti2Function, computeCarryingCapacity, rasterizeFunction, niftiPointwiseObservationOp
    from dt4co.qoi import computeDice, computeTTV, computeVoxelCCC, computeTTC, computeVoxelDice
    from dt4co.utils.fenics_io import read_mv_from_h5
    from dt4co.utils.parallel import root_print
    
    # ------------------------------------------------------------
    # General setup.
    # ------------------------------------------------------------
    SEP = "\n"+"#"*80+"\n"
    SUFFIX = f"_data"  # for the HDF5 data files.
    
    # MPI setup.
    COMM = dl.MPI.comm_world
    rank = COMM.rank
    dl.set_log_level(dl.LogLevel.WARNING)  # suppress FEniCS output.
    
    root_print(COMM, SEP)
    
    # Paths for data.
    MESH_FPATH = args.mesh
    OUT_DIR = args.outdir
    PATIENT_DIR = args.pdir
    
    SAMPLE_TYPE = args.sample_type
    THRESHOLD = args.threshold
    
    SAMPLES_FILE = args.samples
    NSAMPLES = args.nsamples
    PRED_DATE = args.pred_date  # how long to predict after the last observation [days]

    # make directory for QoI data (rasterized NIfTIs)
    QOI_DIR = os.path.join(OUT_DIR, f"qoi_{SAMPLE_TYPE}")
    os.makedirs(QOI_DIR, exist_ok=True)

    # which quantities of interest to compute.
    DO_DICE = args.dice
    DO_TTV = args.ttv
    DO_VOXQOI = args.vox
    DO_TTC = args.ttc

    # ------------------------------------------------------------
    # Set up the experiment.
    # ------------------------------------------------------------
    
    # Get the physical dimension, define mesh functions.
    mesh = load_mesh(COMM, MESH_FPATH)
    report_mesh_info(mesh)
    
    exp = synthExperiment()
    
    root_print(COMM, "Setting the function spaces.")
    Vh = exp.setupBIPFunctionSpaces(mesh, mle=False)
    
    # ------------------------------------------------------------
    # Read back states from file.
    # ------------------------------------------------------------
    
    root_print(COMM, f"Reading back data...")
    root_print(COMM, SEP)
    
    helpfun = dl.Function(Vh[hp.STATE])
    umv = hp.MultiVector(helpfun.vector(), NSAMPLES)
    start = time.perf_counter()
    read_mv_from_h5(COMM, umv, Vh[hp.STATE], SAMPLES_FILE, name="state")
    ttime = time.perf_counter() - start
    root_print(COMM, f"Time to read data: {ttime:.2f} seconds.")
    
    # ------------------------------------------------------------
    # Compute QoIs.
    # ------------------------------------------------------------
    
    # set up the reference image
    # REF_NII = os.path.join(PATIENT_DIR, f"synthetic_noisy_obs_day_{int(PRED_DATE):03d}.nii")
    REF_NII = os.path.join(PATIENT_DIR, f"synthetic_obs_day_{int(PRED_DATE):03d}.nii")
    ROI_NII = REF_NII  # use the same ROI as the reference image
    reffun = dl.Function(Vh[hp.STATE])
    nifti2Function(REF_NII, reffun, Vh[hp.STATE])
    
    if DO_DICE:
        root_print(COMM, "Computing Dice coefficient...")
        start = time.perf_counter()
        DICE = np.zeros((NSAMPLES, 1))
        for i in range(NSAMPLES):
            helpfun.vector().zero()
            helpfun.vector().axpy(1., umv[i])
            DICE[i] = computeDice(helpfun, reffun, threshold=THRESHOLD)
        end = time.perf_counter()
        root_print(COMM, f"Time to compute Dice coefficient for all data: {end-start:.2f} seconds.")
        
        if rank == 0:
            np.save(os.path.join(OUT_DIR, f"{SAMPLE_TYPE}_dice.npy"), DICE)
        
    if DO_VOXQOI:
        start = time.perf_counter()
        root_print(COMM, "Computing voxel-wise CCC and DICE...")
        CCC = np.zeros((NSAMPLES, 1))
        voxDICE = np.zeros((NSAMPLES, 1))
        
        obsOp = niftiPointwiseObservationOp(REF_NII, Vh[hp.STATE])
        
        for i in range(NSAMPLES):
            RASTER_FILE = os.path.join(QOI_DIR, f"raster_{i:06d}.nii")
            helpfun.vector().zero()
            helpfun.vector().axpy(1., umv[i])
            
            # only rasterize if necessary.
            if not os.path.isfile(RASTER_FILE):
                rasterizeFunction(helpfun, Vh[hp.STATE], REF_NII, RASTER_FILE, obsOp=obsOp)
            
            if rank == 0:
                # need to run these in serial so that the NIfTI read is not parallelized.
                # need to add to array when in parallel mode.
                CCC[i] = computeVoxelCCC(RASTER_FILE, REF_NII, ROI_NII)
                voxDICE[i] = computeVoxelDice(RASTER_FILE, REF_NII, threshold=THRESHOLD)
                                
        end = time.perf_counter()
        root_print(COMM, f"Time to compute CCC for all data: {end-start:.2f} seconds.")
        
        if rank == 0:
            np.save(os.path.join(OUT_DIR, f"{SAMPLE_TYPE}_ccc.npy"), CCC)
            np.save(os.path.join(OUT_DIR, f"{SAMPLE_TYPE}_voxdice.npy"), voxDICE)
    
    if DO_TTC:
        root_print(COMM, "Computing TTC...")
        start = time.perf_counter()
        carry_cap = computeCarryingCapacity(REF_NII)
        TTC_true = computeTTC(reffun, carry_cap, threshold=THRESHOLD)  # true value
        TTC = np.zeros((NSAMPLES, 1))
        for i in range(NSAMPLES):
            helpfun.vector().zero()
            helpfun.vector().axpy(1., umv[i])
            TTC[i] = computeTTC(helpfun, carry_cap, threshold=THRESHOLD)
        end = time.perf_counter()
        root_print(COMM, f"Time to compute TTC for all data: {end-start:.2f} seconds.")
        
        if rank == 0:
            np.save(os.path.join(OUT_DIR, f"{SAMPLE_TYPE}_ttc.npy"), TTC)
        
    if DO_TTV:
        start = time.perf_counter()
        root_print(COMM, "Computing normalized tumor volume...")
        TTV = np.zeros((NSAMPLES, 1))
        TTV_true = computeTTV(reffun, threshold=THRESHOLD)  # true value
        for i in range(NSAMPLES):
            helpfun.vector().zero()
            helpfun.vector().axpy(1., umv[i])
            TTV[i] = computeTTV(helpfun, threshold=THRESHOLD)
        end = time.perf_counter()
        root_print(COMM, f"Time to compute TTV for all data: {end-start:.2f} seconds.")
        
        if rank == 0:
            np.save(os.path.join(OUT_DIR, f"{SAMPLE_TYPE}_ttv.npy"), TTV)
            np.save(os.path.join(OUT_DIR, f"ttv_true.npy"), TTV_true)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute quantities of interest.")
    
    # data inputs.
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--pdir", type=str, help="Path to the patient data directory.")
    
    # propagation inputs.
    parser.add_argument("--samples", type=str, required=True, help="Full path to the samples data file.")
    parser.add_argument("--nsamples", type=int, required=True, help="Number of samples to use.")
    parser.add_argument("--sample_type", type=str, required=True, choices=["prior", "la_post", "mcmc", "map", "prior_mean"], help="Type of samples file.")
    parser.add_argument("--pred_date", type=int, required=True, help="How many days to predict in the future?")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold for the tumor.")
    
    # output options.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    
    # Which quantities of interest to compute.
    parser.add_argument("--dice", action=argparse.BooleanOptionalAction, default=True, help="Compute the Dice coefficient.")
    parser.add_argument("--ttv", action=argparse.BooleanOptionalAction, default=True, help="Compute the total tumor volume.")
    parser.add_argument("--vox", action=argparse.BooleanOptionalAction, default=True, help="Compute the voxel-wise QoIs.")
    parser.add_argument("--ttc", action=argparse.BooleanOptionalAction, default=True, help="Compute the time to threshold.")
    
    # Parse the arguments, strip CLI args for PETSc.
    args = parser.parse_args()
    
    main(args)
