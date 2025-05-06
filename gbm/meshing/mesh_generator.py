###############################################################################
# 
# This script generates a mesh of the brain given processed FreeSurfer segmentations.
# 
# NOTE:
#   If you are generating a 2D mesh, make sure that you are setting the z-offset correctly.
#       If the surface has been shifted by the FreeSurfere C_{RAS}, you
#       will need to adjust the z-offset accordingly.
#       Recall the plane equation is given by ax + by + cz + d = 0.
#       The z-offset is the value of d (negative values move you in the superior direction).
# 
# 
# Example usage: python3 meshing/mesh_generator.py --stlpath /path/to/stl/ --outdir /path/to/store
# 
###############################################################################

import sys
import os
import argparse
import time

import dolfin as dl
from mpi4py import MPI  # MUST be imported AFTER dolfin

sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
from dt4co.utils.mesh_utils import meshio_to_dolfin_all, write_xdmf_to_h5, write_h5_to_xdmf, report_mesh_info
from dt4co.utils.svmtk import svmtk_fullbrain_five_domain, svmtk_create_gwv_mesh
from dt4co.utils.refine_svmtk import svmtk_refine


def main(args):
    ############################################################
    # 0. Build necessary filenames.
    ############################################################
    SEP="\n"+"*"*80+"\n"  # separator for printing
    MESHBASE = args.fname
    os.makedirs(args.outdir, exist_ok=True)  # make sure output directory exists.
    XDMFMESH = os.path.join(args.outdir, f"{MESHBASE}.xdmf")
    SUBDOMAINMESH = os.path.join(args.outdir, f"{MESHBASE}-subdomains.xdmf")
    BNDRYMESH = os.path.join(args.outdir, f"{MESHBASE}-boundaries.xdmf")
    STLPATH = args.stlpath
    H5FILE = os.path.join(args.outdir, f"{MESHBASE}-all.h5")
    
    COMM = dl.MPI.comm_world
    
    # Handle 2D case.
    assert args.gdim in [2, 3], "Geometric dimension must be 2 or 3."
    if args.gdim == 2:
        print("Mesh will be 2-dimensional.")
        PLANE = 0, 0, 1, args.zoff
        MSHMESH = os.path.join(args.outdir, f"{MESHBASE}.mesh")
    else:
        print("Mesh will be 3-dimensional.")
        PLANE = None
        MSHMESH = os.path.join(args.outdir, f"{MESHBASE}.mesh")
    
    print(f"Ventricles will be removed: {args.remove_ventricles}")
    
    print(SEP)
    print(f"Using STL files from:\t\t{STLPATH}")
    print(f"Mesh will be saved to:\t\t{XDMFMESH}")
    print(f"Subdomains will be saved to:\t{SUBDOMAINMESH}")
    print(f"Boundaries will be saved to:\t{BNDRYMESH}")
    print(f"Supporting HDF5 file will be saved to:\t{H5FILE}")
    print(SEP)
        
    ############################################################
    # 1. Mesh the domain using SVM-Tk.
    ############################################################
    start = time.perf_counter()
    if (args.hemi == "") or (args.hemi == "both"):
        print(f"Meshing BOTH hemispheres.")
        # Call five domain meshing function.
        svmtk_fullbrain_five_domain(
            stls=[
                os.path.join(STLPATH, f"lh.pial{args.stlmod}.stl"),
                os.path.join(STLPATH, f"rh.pial{args.stlmod}.stl"),
                os.path.join(STLPATH, f"lh.white{args.stlmod}.stl"),
                os.path.join(STLPATH, f"rh.white{args.stlmod}.stl"),
                os.path.join(STLPATH, f"ventricles{args.stlmod}.stl"),
                ],
            output=MSHMESH,
            resolution=args.resolution,
            remove_ventricles=args.remove_ventricles,
            plane=PLANE
        )
        
    elif args.hemi == "lh":
        print(f"Meshing LEFT hemisphere ONLY.")
        svmtk_create_gwv_mesh(
            pialstl=os.path.join(STLPATH, f"lh.pial{args.stlmod}.stl"),
            whitestl=os.path.join(STLPATH, f"lh.white{args.stlmod}.stl"),
            ventstl=os.path.join(STLPATH, f"ventricles{args.stlmod}.stl"),
            output=MSHMESH,
            resolution=args.resolution,
            remove_ventricles=args.remove_ventricles,
            plane=PLANE
        )
    
    elif args.hemi == "rh":
        print(f"Meshing RIGHT hemisphere ONLY.")
        svmtk_create_gwv_mesh(
            pialstl=os.path.join(STLPATH, f"rh.pial{args.stlmod}.stl"),
            whitestl=os.path.join(STLPATH, f"rh.white{args.stlmod}.stl"),
            ventstl=os.path.join(STLPATH, f"ventricles{args.stlmod}.stl"),
            output=MSHMESH,
            resolution=args.resolution,
            remove_ventricles=args.remove_ventricles,
            plane=PLANE
        )
        
    else:
        raise ValueError(f"Invalid hemisphere specified: {args.hemi}")
        
    print(f"Meshing took {(time.perf_counter()-start) / 60.:.2f} minutes.")
    print(SEP)

    ############################################################
    # 2. Convert the meshio format to doflin readable XDMF.
    ############################################################
    print("Writing dolfin-readable XDMF mesh file.")
    meshio_to_dolfin_all(MSHMESH, XDMFMESH, SUBDOMAINMESH, BNDRYMESH)
    print("Writing HDF5 file with mesh, subdomain, boundary.")
    write_xdmf_to_h5(XDMFMESH, SUBDOMAINMESH, BNDRYMESH, H5FILE)
    print(SEP)
    
    ############################################################
    # 3. Optionally print some information about the mesh.
    ############################################################
    if args.print_info:
        mesh = dl.Mesh()
        with dl.XDMFFile(COMM, XDMFMESH) as fid:
            fid.read(mesh)
        
        report_mesh_info(mesh)
    
    if args.refine:
        REFINED_BASE = f"{MESHBASE}-refined"
        REFINEDMESH = os.path.join(args.outdir, f"{REFINED_BASE}-all.h5")
        svmtk_refine(mesh=XDMFMESH, subdomains=SUBDOMAINMESH, boundaries=BNDRYMESH, outfname=REFINEDMESH)
        
        write_h5_to_xdmf(REFINEDMESH, os.path.join(args.outdir, REFINED_BASE))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a mesh of the brain given processed FreeSurfer segmentations.")
    
    # Required arguments.
    parser.add_argument("--stlpath", type=str, help="Path to directory containing STL files.")
    parser.add_argument("--outdir", type=str, help="Output directory for mesh files.")
    
    # Output options.
    parser.add_argument("--fname", type=str, default="mesh", help="Base filename for mesh files.")
    parser.add_argument("--print-info", action=argparse.BooleanOptionalAction, default=True, help="Print mesh info after generation. Default is True.")
        
    # Options for meshing.
    parser.add_argument("--resolution", type=int, default=32, help="Resolution of the mesh.")
    parser.add_argument("--hemi", type=str, default="", help="Which hemisphere should be meshed. An empty string '' means to mesh both hemispheres. Use 'lh' or 'rh' to specify an individual hemisphere.")
    parser.add_argument("--remove-ventricles", action=argparse.BooleanOptionalAction, default=True, help="Keep ventricles in the brain mesh. Default is to remove ventricles.")
    parser.add_argument("--stlmod", type=str, default=".shifted", help="Modifier to append to STL filenames, stemming from transform applied to FreeSurfer surfaces.")
    parser.add_argument("--gdim", type=int, default=3, help="Geometric dimension of the mesh. Default is 3.")
    parser.add_argument("--zoff", type=float, default=0., help="Z-coord for mesh slice. Default is 0.")
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction, default=False, help="Refine the mesh after generation. Default is False.")

    args = parser.parse_args()
    main(args)
