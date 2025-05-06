################################################################################
# This script post-processes the output of the longitudinal registration pipeline.
# In particular:
#   - ADC and ROI images are combined to generate a cellularity map.
#   - Optionally, the cellularity map and the ROI are projected onto a mesh for visualization.
#   - Optionally, the deformation field is applied to the mesh for visualization.
# 
# Usage: python3 viz_registration.py --pinfo /path/to/patient/info
#                   --datadir /path/to/patient/data          
#                   --mesh /path/to/mesh/
#                   --outdir /path/to/store/output
# 
################################################################################

import os
import sys
import argparse

import dolfin as dl
from mpi4py import MPI  # MUST be imported AFTER dolfin

sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
from dt4co.utils.data_utils import nifti2Function, vectorNifti2Function
from dt4co.utils.mesh_utils import report_mesh_info, check_mesh_dimension
from dt4co.dataModel import PatientData

def main(args)->None:
    
    # ---------------------------------------------------------------
    # Unpack arguments and build necessary paths.
    # ---------------------------------------------------------------
    
    COMM = MPI.COMM_WORLD  # MPI communicator
    
    pinfo = PatientData(args.pinfo, args.datadir)
    
    MESHFILE = args.mesh
    OUTPUT_DIR = args.outdir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Patient data directory: {args.datadir}")
    print(f"Found {pinfo.get_num_visits()} imaging dates.")
    print(f"Imaging dates:")
    pinfo.print_timeline()
    
    # ---------------------------------------------------------------
    # Write the cellularity maps + ROIs to file for visualization.
    # ---------------------------------------------------------------
    
    mesh = dl.Mesh()
    with dl.XDMFFile(COMM, MESHFILE) as fid:
        fid.read(mesh)
    
    ZOFF = check_mesh_dimension(mesh, args.zoff)
    
    report_mesh_info(mesh)
    
    # Create function spaces, write the segmentations to file.
    Vh = dl.FunctionSpace(mesh, "Lagrange", 2)
    ufun = dl.Function(Vh, name="roi")
    
    ROI_VIZ_FILE = os.path.join(OUTPUT_DIR, "viz_roi.xdmf")
    print(f"Writing ROI data to file:\t{ROI_VIZ_FILE}")
    with dl.XDMFFile(COMM, ROI_VIZ_FILE) as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
        for i, visit in enumerate(pinfo.visits):
            ufun.vector().zero()
            nifti2Function(str(visit.roi), ufun, Vh, zoff=ZOFF, roi=True)
            fid.write(ufun, i)
    
    ufun = dl.Function(Vh, name="tumor")
    TUMOR_VIZ_FILE = os.path.join(OUTPUT_DIR, "viz_tumor.xdmf")
    print(f"Writing tumor data to file:\t{TUMOR_VIZ_FILE}")
    with dl.XDMFFile(COMM, TUMOR_VIZ_FILE) as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
        for i, visit in enumerate(pinfo.visits):
            ufun.vector().zero()
            nifti2Function(str(visit.tumor), ufun, Vh, zoff=ZOFF)
            fid.write(ufun, i)
    
    # ---------------------------------------------------------------
    # (Optionally) Write the deformations to file for visualization.
    # ---------------------------------------------------------------
    
    if args.deformations:
        
        Vh = dl.VectorFunctionSpace(mesh, "Lagrange", 2)
        ufun = dl.Function(Vh, name="deformation")

        DEFORMATION_VIZ_FILE = os.path.join(OUTPUT_DIR, "viz_deformation.xdmf")
        print(f"Writing deformation data to file:\t{DEFORMATION_VIZ_FILE}")
        with dl.XDMFFile(COMM, DEFORMATION_VIZ_FILE) as fid:
            fid.parameters["functions_share_mesh"] = True
            fid.parameters["rewrite_function_mesh"] = False
            for i, visit in enumerate(pinfo.visits):
                ufun.vector().zero()
                if i == 0:
                    fid.write(ufun, i)  # no deformation for the first visit
                else:
                    vectorNifti2Function(str(visit.deformation), ufun, Vh)
                    fid.write(ufun, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the tumor cellularity maps and write them to file.")
    
    parser.add_argument("--pinfo", type=str, help="Path to the patient information file.")
    parser.add_argument("--datadir", type=str, help="Path to the patient data directory.")
    
    # Optional arguments, for visualization.
    parser.add_argument("--mesh", type=str, help="Path to the mesh file.")
    parser.add_argument("--outdir", type=str, help="Output directory for visualization.")

    # Input options.
    parser.add_argument("--zoff", type=float, default=None, help="Z-offset for 2D meshes.")
    parser.add_argument("--deformations", action=argparse.BooleanOptionalAction, default=False, help="Apply the deformations to the mesh.")

    args = parser.parse_args()
    
    main(args)