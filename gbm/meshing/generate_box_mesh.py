import os
import sys
import argparse

import numpy as np
from mpi4py import MPI
import dolfin as dl

sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
from dt4co.utils.mesh_utils import get_mesh_bbox, report_mesh_info

def main(args)->None:
    
    MESHFILE = args.mesh   # get the path to the mesh file.
    OUT = os.path.join(args.outdir, f"{args.fname}.xdmf")
    
    # Load mesh.
    mesh = dl.Mesh()
    with dl.XDMFFile(MPI.COMM_WORLD, MESHFILE) as fid:
        fid.read(mesh)
    
    # Create ~ 1mm isotropic mesh from bounding box.
    bounds = get_mesh_bbox(mesh)
    pmin = dl.Point(bounds[0, :])
    pmax = dl.Point(bounds[1, :])
    ncells  = np.rint(bounds[1,:] - bounds[0,:]).astype(int)
    
    # Generate the mesh and write it to file.
    if mesh.topology().dim() == 2:
        box = dl.RectangleMesh(pmin, pmax, *ncells)
        with dl.XDMFFile(MPI.COMM_WORLD, OUT) as fid:
            fid.write(box)
    elif mesh.topology().dim() == 3:
        box = dl.BoxMesh(pmin, pmax, *ncells)
        with dl.XDMFFile(MPI.COMM_WORLD, OUT) as fid:
            fid.write(box)
    
    # Give some information about the mesh.
    report_mesh_info(box)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a box mesh from a given mesh file.")
    
    # Required arguments.
    parser.add_argument("--mesh", type=str, default=None, help="Path to mesh file from which to generate the bounding box mesh.")
    parser.add_argument("--outdir", type=str, help="Output directory for mesh files.")
    
    # Output options.
    parser.add_argument("--fname", type=str, default="box", help="Base filename for mesh files.")

    args = parser.parse_args()
        
    main(args)
