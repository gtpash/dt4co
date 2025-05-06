################################################################################
# 
# This script concatenates forward propagation data for each visit. It reads back the state data from the forward propagation.
# 
# An example call to this script is:
# mpirun -np <num_procs> python3 concat_fwd_prop.py
#        --mesh /path/to/mesh/
#        --pinfo /path/to/patient_info.json
#        --pdir /path/to/patient/data/
#        --datadir /path/to/prop/data/
#        --experiment_type experiment_type
#        --outdir /path/to/store/results/
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
    from dt4co.utils.fenics_io import write_mv_to_h5, write_mv_to_xdmf, read_mv_from_h5
    from dt4co.utils.parallel import root_print
    from dt4co.dataModel import PatientData
    from dt4co.experiments import ExperimentFactory
    
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
    DATA_DIR = args.datadir
    IS_L2F = args.l2f

    # Load in the patient data.
    pinfo = PatientData(args.pinfo, args.pdir)

    # Experiment setup.
    factory = ExperimentFactory(pinfo)
    exp = factory.get_experiment(args.experiment_type)
    root_print(COMM, f"Using experiment: {args.experiment_type}")
    root_print(COMM, f"Experiment instance: {type(exp)}")

    # ------------------------------------------------------------
    # Set up the experiment.
    # ------------------------------------------------------------
    
    root_print(COMM, f"Loading in the mesh...")
    root_print(COMM, SEP)
    
    mesh, _ = exp.setupMesh(COMM, MESH_FPATH, zoff=None)
    Vh = exp.setupFunctionSpaces(mesh, mle=False)
    
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
    
    
    data_days = pinfo.visit_days if not IS_L2F else [ pinfo.visit_days[-1] ]
    
    for vidx, vday in enumerate(data_days):
        root_print(COMM, f"Processing visit {vidx}: day {vday}")
        
        # zero out the data
        umv_out.zero()
        DATA_NAME = f"data/state/{vidx:06d}"
        
        for idx, f in enumerate(data_files):
            helpmv.zero()                      # zero out the MultiVector for fresh data.
            fpath = os.path.join(DATA_DIR, f)
            read_mv_from_h5(COMM, helpmv, Vh[hp.STATE], fpath, name=[DATA_NAME])
            umv_out[idx].axpy(1., helpmv[0])  # copy the function data to the MultiVector.
            
        root_print(COMM, f"Writing concatenated data to file...")
        ALL_VIZ = os.path.join(OUT_DIR, f"{PREFIX}v{vidx}_state_all.xdmf")
        ALL_DATA = os.path.join(OUT_DIR, f"{PREFIX}v{vidx}_all{SUFFIX}.h5")
        
        write_mv_to_xdmf(COMM, umv_out, Vh[hp.STATE], ALL_VIZ, name="state")
        write_mv_to_h5(COMM, umv_out, Vh[hp.STATE], ALL_DATA, name="state")
        
        root_print(COMM, f"Concatenated state (visualization) written to:\t{ALL_VIZ}")
        root_print(COMM, f"Concatenated state (data) written to:\t{ALL_DATA}")
        root_print(COMM, SEP)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BIP for the synthetic experiment.")
    
    # data inputs.
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--pinfo", type=str, help="Path to the patient information file.")
    parser.add_argument("--pdir", type=str, help="Path to the patient data directory.")
    parser.add_argument("--datadir", type=str, required=True, help="Path to the data files.")
    
    # modeling inputs.
    parser.add_argument("--experiment_type", type=str, required=True, choices=["rd", "rdtx", "pwrdtx"], help="Type of experiment to run.")
    parser.add_argument("--l2f", action=argparse.BooleanOptionalAction, default=True, help="Make prediction from last observation to holdout?")
    
    # output options.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--prefix", type=str, required=False, default="bip", help="Name prefix for the output files.")
    
    # Parse the arguments, strip CLI args for PETSc.
    args, other = parser.parse_known_args()
    petsc4py.init(other)
    
    main(args)
