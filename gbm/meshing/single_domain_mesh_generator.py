############################################################
# For meshing a single pial hemisphere, either in 2D or 3D
############################################################

import sys
import os
import argparse
import time

from mpi4py import MPI
import dolfin as dl
import meshio

sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
from dt4co.utils.mesh_utils import create_dolfin_mesh, report_mesh_info, load_mesh
from dt4co.utils.svmtk import svmtk_create_slice_mesh, svmtk_create_volume_mesh

def main(args):
    ############################################################
    # 0. Build necessary filenames.
    ############################################################
    SEP="\n"+"*"*80+"\n"  # separator for printing
    MESHBASE = args.fname
    os.makedirs(args.outdir, exist_ok=True)  # make sure output directory exists.
    MSHMESH = os.path.join(args.outdir, f"{MESHBASE}.mesh")
    XDMFMESH = os.path.join(args.outdir, f"{MESHBASE}.xdmf")
    STLPATH = args.stlpath
    
    print(SEP)
    print(f"Using STL files from:\t\t{STLPATH}")
    print(f"Mesh will be saved to:\t\t{XDMFMESH}")
    print(SEP)
        
    ############################################################
    # 1. Mesh the domain using SVM-Tk.
    ############################################################
    if args.hemi == "lh":
        print(f"Meshing left hemisphere ONLY.")
        stl = os.path.join(STLPATH, f"lh.pial{args.stlmod}.stl")
    elif args.hemi == "rh":
        print(f"Meshing right hemisphere ONLY.")
        stl = os.path.join(STLPATH, f"rh.pial{args.stlmod}.stl")
    else:
        raise ValueError(f"Invalid hemisphere specified: {args.hemi}")
    
    start = time.perf_counter()
    assert args.gdim in [2, 3], "Geometric dimension must be 2 or 3."
    if args.gdim == 2:
        print("Mesh will be 2-dimensional.")
        PLANE = 0, 0, 1, args.zoff
        svmtk_create_slice_mesh(
            stlfile=stl,
            output=MSHMESH,
            plane=PLANE,
            resolution=args.resolution
        )
        
    else:
        print("Mesh will be 3-dimensional.")
        # call the appropriate SVMTK meshing function.
        svmtk_create_volume_mesh(
            stlfile=stl,
            output=MSHMESH,
            resolution=args.resolution
        )
        
    print(f"Meshing took {(time.perf_counter()-start) / 60.:.2f} minutes.")
    print(SEP)
    
    ############################################################
    # 2. Convert the meshio format to doflin readable XDMF.
    ############################################################
    print("Writing dolfin-readable XDMF mesh file.")
    meshio_mesh = meshio.read(MSHMESH)
    if args.gdim == 2:
        tmp = meshio.read(MSHMESH)
        meshio.write(XDMFMESH, tmp)
        if args.prunez:
            PRUNEMESH = os.path.join(args.outdir, f"{MESHBASE}_2d.xdmf")
            tmp.points = tmp.points[:, :2]  # prune z
            meshio.write(PRUNEMESH, tmp)
    else:
        xdmf_mesh = create_dolfin_mesh(meshio_mesh, cell_type="tetra", prune_z=False)
        meshio.write(XDMFMESH, xdmf_mesh)
    
    print(SEP)
    
    ############################################################
    # 3. Optionally print some information about the mesh.
    ############################################################
    if args.print_info:
        mesh = load_mesh(MPI.COMM_WORLD, XDMFMESH)    
        report_mesh_info(mesh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a mesh of the brain given processed FreeSurfer segmentations.")
    
    # Required arguments.
    parser.add_argument("--stlpath", type=str, help="Path to directory containing STL files.")
    parser.add_argument("--outdir", type=str, help="Output directory for mesh files.")
    
    # Output options.
    parser.add_argument("--fname", type=str, default="mesh", help="Base filename for mesh files.")
    parser.add_argument("--print-info", action=argparse.BooleanOptionalAction, default=True, help="Print mesh info after generation. Default is True.")
    parser.add_argument("--prunez", action=argparse.BooleanOptionalAction, default=True, help="Save a copy of the mesh with the z-coordinates pruned. This is primarily for inspecting mesh quality. Default is True.")
        
    # Options for meshing.
    parser.add_argument("--resolution", type=int, default=32, help="Resolution of the mesh.")
    parser.add_argument("--hemi", type=str, help="Which hemisphere should be meshed. Use 'lh' or 'rh' to specify left/right respectively.")
    parser.add_argument("--stlmod", type=str, default=".shifted", help="Modifier to append to STL filenames, stemming from transform applied to FreeSurfer surfaces.")
    parser.add_argument("--gdim", type=int, default=3, help="Geometric dimension of the mesh. Default is 3.")
    parser.add_argument("--zoff", type=float, default=0., help="Z-coord for mesh slice. Default is 0.")

    args = parser.parse_args()
    main(args)
