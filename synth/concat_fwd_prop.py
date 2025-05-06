################################################################################
# 
# This script concatenates forward propagation data for each visit. It reads back the state data from the forward propagation.
# 
# An example call to this script is:
# mpirun -np <num_procs> python3 concat_fwd_prop.py
#        --mesh /path/to/mesh/
#        --datadir /path/to/prop/data/
#        --experiment_type experiment_type
#        --outdir /path/to/store/results/
#        --imgfreq imaging_frequency
#        --l2f last_to_final flag
#        --pred_date prediction_date
#        --prefix prefix for the output data
# 
################################################################################

import os
import sys
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
    from dt4co.utils.fenics_io import write_mv_to_h5, write_mv_to_xdmf, read_mv_from_h5
    from dt4co.utils.parallel import root_print
    
    # ------------------------------------------------------------
    # General setup.
    # ------------------------------------------------------------
    SEP = "\n"+"#"*80+"\n"
    
    PREFIX = f"{args.prefix}_" if args.prefix is not None else ""
    SUFFIX = f"_data"  # for the HDF5 data files.
    
    dl.set_log_level(dl.LogLevel.WARNING)  # suppress FEniCS output.
    
    # MPI setup.
    COMM = dl.MPI.comm_world
    root_print(COMM, SEP)
    
    # Paths for data.
    MESH_FPATH = args.mesh
    OUT_DIR = args.outdir
    
    EXP_TYPE = args.experiment_type
    IS_L2F = args.l2f
    DATA_DIR = args.datadir

    IMG_FREQ = args.imgfreq     # imaging frequency [days] (every third day)
    IMG_FREQ = 1 if IS_L2F else IMG_FREQ  # set up proper last observation if doing last-to-final
    PRED_DATE = args.pred_date  # how long to predict after the last observation [days]
    TX_START = 14.0             # start of therapy [days]

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
    
    visit_days = [img_days[-1]] if IS_L2F else img_days
    
    # ------------------------------------------------------------
    # Read back states and concatenate data.
    # ------------------------------------------------------------
    
    root_print(COMM, f"Reading back data to concatenate...")
    
    # scan the directory to get the data files.
    data_files = [f.name for f in os.scandir(DATA_DIR) if SUFFIX in f.name]
    
    root_print(COMM, f"Found {len(data_files)} data files.")
    root_print(COMM, SEP)
    
    helpfun = dl.Function(Vh[hp.STATE])
    helpmv = hp.MultiVector(helpfun.vector(), 1)
    umv_out = hp.MultiVector(helpfun.vector(), len(data_files))
    
    for vidx, vday in enumerate(visit_days):
        root_print(COMM, f"Processing visit {vidx}: day {visit_days[vidx]}")
        
        # zero out the data
        umv_out.zero()
        DATA_NAME = f"data/state/{vidx:06d}"
        
        for idx, f in enumerate(data_files):
            helpmv.zero()                      # zero out the MultiVector for fresh data.
            fpath = os.path.join(DATA_DIR, f)
            read_mv_from_h5(COMM, helpmv, Vh[hp.STATE], fpath, name=[DATA_NAME])
            umv_out[idx].axpy(1., helpmv[0])  # copy the function data to the MultiVector.
            
        root_print(COMM, f"Writing concatenated data to file...")
        ALL_VIZ = os.path.join(OUT_DIR, f"{PREFIX}day{int(vday):03d}_all.xdmf")
        ALL_DATA = os.path.join(OUT_DIR, f"{PREFIX}day{int(vday):03d}_all{SUFFIX}.h5")
        
        write_mv_to_xdmf(COMM, umv_out, Vh[hp.STATE], ALL_VIZ, name="state")
        write_mv_to_h5(COMM, umv_out, Vh[hp.STATE], ALL_DATA, name="state")
        
        root_print(COMM, f"Concatenated state (visualization) written to:\t{ALL_VIZ}")
        root_print(COMM, f"Concatenated state (data) written to:\t{ALL_DATA}")
        root_print(COMM, SEP)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BIP for the synthetic experiment.")
    
    # data inputs.
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--imgfreq", type=int, required=False, default=1, help="Frequency of images [days].")
    parser.add_argument("--datadir", type=str, required=True, help="Path to the data files.")
    
    # modeling inputs.
    parser.add_argument("--experiment_type", type=str, required=True, choices=["rd", "rdtx"], help="Type of experiment to run.")
    
    # propagation options.
    parser.add_argument("--l2f", action=argparse.BooleanOptionalAction, default=True, help="Make prediction from last observation to holdout?")
    parser.add_argument("--pred_date", type=int, required=True, default=7, help="How many days to predict in the future?")
    
    # output options.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--prefix", type=str, required=False, default="bip", help="Name prefix for the output files.")
    
    # Parse the arguments, strip CLI args for PETSc.
    args, other = parser.parse_known_args()
    petsc4py.init(other)
    
    main(args)
