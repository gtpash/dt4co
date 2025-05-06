import os
import sys
import time
import argparse

import dolfin as dl
import ufl

sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
from dt4co.utils.refine_svmtk import svmtk_refine
from dt4co.utils.mesh_utils import report_mesh_info
from dt4co.utils.parallel import root_print


def main(args):
    SEP="\n"+"*"*80+"\n"
    
    # Build necessary filenames.
    MESHFPATH = args.mesh
    FNAME = args.fname
    OUTDIR = args.outdir
    OUTMESH = os.path.join(OUTDIR, f"{FNAME}.h5")
    
    COMM = dl.MPI.comm_world
    
    root_print(COMM, SEP)
    root_print(COMM, f"Refining provided mesh:\t{MESHFPATH}")
    root_print(COMM, SEP)
    
    # XDMFMESH = os.path.join(args.mesh_dir, MESHBASE+".xdmf")
    # SUBDOMAINMESH = os.path.join(args.mesh_dir, MESHBASE+"-subdomains.xdmf")
    # BNDRYMESH = os.path.join(args.mesh_dir, MESHBASE+"-boundaries.xdmf")
    # HDF5MESH = os.path.join(args.mesh_dir, MESHBASE+"-all.h5")
    # REFINEDMESH = os.path.join(args.mesh_dir, MESHBASE+"-refined.h5")
    
    # Read in the subdomains and boundaries.
    mesh = dl.Mesh()
    hdf = dl.HDF5File(mesh.mpi_comm(), MESHFPATH, "r")
    hdf.read(mesh, "/mesh", False)
    
    # Read subdomains and boundary markers
    d = mesh.topology().dim()
    subdomains = dl.MeshFunction("size_t", mesh, d)
    hdf.read(subdomains, "/subdomains")
    boundaries = dl.MeshFunction("size_t", mesh, d-1)
    hdf.read(boundaries, "/boundaries")
    hdf.close()
    # else:
    #     print("No HDF5 mesh found, falling back to individual XDMF files.")
    #     print(f"Could not find the HDF5 mesh:\t{HDF5MESH}")
    #     print(f"Reading mesh from:\t\t{XDMFMESH}")
    #     print(f"Reading subdomains from:\t{SUBDOMAINMESH}")
    #     print(f"Reading boundaries from:\t{BNDRYMESH}")
        
    #     # Read mesh, subdomains, and boundaries.
    #     with dl.XDMFFile(mesh.mpi_comm(), XDMFMESH) as fid:
    #         fid.read(mesh)

    #     d = mesh.topology().dim()

    #     with dl.XDMFFile(mesh.mpi_comm(), SUBDOMAINMESH) as fid:
    #         subdomains = dl.MeshFunction("size_t", mesh, d)
    #         fid.read(subdomains, "subdomains")
            
    #     with dl.XDMFFile(mesh.mpi_comm(), BNDRYMESH) as fid:
    #         boundaries = dl.MeshFunction("size_t", mesh, d-1)
    #         fid.read(boundaries, "boundaries")

    
    report_mesh_info(mesh)
    
    # Perform the mesh refinement.
    root_print(COMM, f"Refining mesh and saving to:\t{OUTMESH}")
    start = time.perf_counter()
    svmtk_refine(mesh=mesh,
                subdomains=subdomains,
                boundaries=boundaries,
                outfname=OUTMESH
    )
    finish = time.perf_counter()
    root_print(COMM, f"Refinement took {(finish-start) / 60.:.2f} minutes.")
    
    refined_mesh = dl.Mesh()
    hdf = dl.HDF5File(refined_mesh.mpi_comm(), OUTMESH, "r")
    hdf.read(refined_mesh, "/mesh", False)
    hdf.close()
    
    root_print(COMM, SEP)
    root_print(COMM, f"Reading back refined mesh...")
    report_mesh_info(refined_mesh)
    
    with dl.XDMFFile(COMM, os.path.join(OUTDIR, f"viz_{FNAME}.xdmf")) as fid:
        fid.write(refined_mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine the mesh.")
    
    # Required arguments.
    parser.add_argument("--mesh", type=str, default=None, help="Path to mesh file from which to generate the bounding box mesh.")
    parser.add_argument("--outdir", type=str, help="Output directory for mesh files.")
    
    # Output options.
    parser.add_argument("--fname", type=str, default="box", help="Base filename for mesh files.")
    
    args = parser.parse_args()
    main(args)

