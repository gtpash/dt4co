import sys
import os
import argparse
import time

from mpi4py import MPI
import dolfin as dl
import meshio

sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
from dt4co.utils.mesh_utils import load_mesh, report_mesh_info
from dt4co.utils.svmtk import svmtk_create_slice_mesh

# NOTE: Make sure that you are setting the z-offset correctly.
#      If the surface has been shifted by the FreeSurfere C_{RAS}, you
#      will need to adjust the z-offset accordingly.
#      Recall the plane equation is given by ax + by + cz + d = 0.
#      The z-offset is the value of d (negative values move you up).

# NOTE: This script currently does not support multiple surfaces, only pial.

def main(args):
    ############################################################
    # 0. Build necessary filenames.
    ############################################################
    SEP="\n"+"*"*80+"\n"  # separator for printing
    MESHBASE = args.fname
    os.makedirs(args.outdir, exist_ok=True)  # make sure output directory exists.
    XDMFMESH = os.path.join(args.outdir, f"{MESHBASE}.xdmf")
    STLPATH = args.stlpath
    
    print("Mesh will be 2-dimensional.")
    PLANE = 0, 0, 1, args.zoff
    MSHMESH = os.path.join(args.outdir, f"{MESHBASE}.stl")
    
    print(SEP)
    print(f"Using STL files from:\t\t{STLPATH}")
    print(f"Mesh will be saved to:\t\t{XDMFMESH}")
    print(SEP)
        
    ############################################################
    # 1. Mesh the domain using SVM-Tk.
    ############################################################
    start = time.perf_counter()
        
    if args.hemi == "lh":
        print(f"Meshing left hemisphere ONLY.")
        svmtk_create_slice_mesh(
            stlfile=os.path.join(STLPATH, f"lh.pial{args.stlmod}.stl"),
            output=MSHMESH,
            plane=PLANE,
            resolution=args.resolution
        )
    
    elif args.hemi == "rh":
        print(f"Meshing right hemisphere ONLY.")
        svmtk_create_slice_mesh(
            stlfile=os.path.join(STLPATH, f"rh.pial{args.stlmod}.stl"),
            output=MSHMESH,
            plane=PLANE,
            resolution=args.resolution
        )
        
    else:
        raise ValueError(f"Invalid hemisphere specified: {args.hemi}")
        
    print(f"Meshing took {(time.perf_counter()-start) / 60.:.2f} minutes.")
    print(SEP)

    ############################################################
    # 2. Convert the meshio format to doflin readable XDMF.
    ############################################################
    print("Writing dolfin-readable XDMF mesh file.")
    tmp = meshio.read(MSHMESH)
    meshio.write(XDMFMESH, tmp)
    
    if args.prunez:
        PRUNEMESH = os.path.join(args.outdir, f"{MESHBASE}_2d.xdmf")
        tmp.points = tmp.points[:, :2]  # prune z
        meshio.write(PRUNEMESH, tmp)
    
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
    parser.add_argument("--hemi", type=str, default="", help="Which hemisphere should be meshed. An empty string '' means to mesh both hemispheres. Use 'lh' or 'rh' to specify an individual hemisphere.")
    parser.add_argument("--stlmod", type=str, default=".shifted", help="Modifier to append to STL filenames, stemming from transform applied to FreeSurfer surfaces.")
    parser.add_argument("--zoff", type=float, default=0., help="Z-coord for mesh slice. Default is 0.")

    args = parser.parse_args()
    main(args)
