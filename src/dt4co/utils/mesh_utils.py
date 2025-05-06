import os
import numpy as np
import meshio
import dolfin as dl
from pathlib import Path

# ----------------------------------------------
# Meshio helpers
# ----------------------------------------------
def gmsh2meshio(mesh, cell_type:str, prune_z:bool=False):
    """Extract `GMSH` mesh and return `meshio` mesh.

    Args:
        mesh: GMSH mesh.
        cell_type (str): Type of mesh cells.
        prune_z (bool, optional): Remove the z-component of the mesh to return a 2D mesh. Defaults to False.

    Returns:
        out_mesh: Converted meshio mesh object.
    """
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:geometrical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh


def create_dolfin_mesh(mesh, cell_type:str, data_name:str="medit:ref", prune_z:bool=False, name_to_read:str="name_to_read", center:bool=False) -> meshio.Mesh:
    """Extract mesh information from a meshio data structure.
    This code is based upon code from J. Dokken, with some modifications.
    ref: https://jsdokken.com/src/pygmsh_tutorial.html

    Args:
        mesh (meshio.Mesh): Meshio mesh.
        cell_type (str): Cell type to extract, e.g. 'tetra' or 'triangle'
        data_name (str, optional): Data name. Defaults to "medit:ref" for use with SVMTk/CGAL. Note: GMSH writes to gmsh:geometrical or gmsh:physical.
        prune_z (bool, optional): Whether or not to prune the 3rd dimension of data. Defaults to False.
        name_to_read (str, optional): New data name to be read later. Defaults to "name_to_read".
        center (bool, optional): Whether or not to center the mesh at the origin.

    Returns:
        mesh (meshio.Mesh): New meshio Mesh object.
    """
    # -----------------------------------
    # One option, using for loops
    
    # cells = np.vstack([cell.data for cell in mesh.cells if cell.type == cell_type])
    # cell_data = np.hstack(
    #     [
    #         mesh.cell_data_dict[data_name][key]
    #         for key in mesh.cell_data_dict[data_name].keys()
    #         if key == cell_type
    #     ]
    # )
    # -----------------------------------
    
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data(data_name, cell_type)
    
    # center, if desired.
    if center:
        centroid = np.mean(mesh.points, axis=0)
    else:
        centroid = 0*np.mean(mesh.points, axis=0)
    
    # prune, or otherwise write points.
    if prune_z:
        points = mesh.points[:, :2] - centroid
    else:
        points = mesh.points - centroid

    out = meshio.Mesh(
        points=points, cells={cell_type: cells}, cell_data={name_to_read: [cell_data]}
    )
    
    return out


def meshio_to_dolfin_all(mio_fpath: str, xdmf_fpath: str, subdomain_fpath: str, boundary_fpath: str)->None:
    """Convert meshio mesh to dolfin-readable XDMF mesh.
    WARNING: This function assumes only triangle and tetrahedral cells are present.
    WARNING: This function assumes that the mesh is 2D or 3
    WARNING: This function will not work for quadrilateral or hexahedral cells.

    Args:
        mio_fpath (str): Absolute path to .mesh or .msh file.
        xdmf_fpath (str): Absolute path for where to save the XDMF mesh file.
        subdomain_fpath (str): Absolute path for where to save the subdomains XMDF file.
        boundary_fpath (str): Absolute path for where to save the boundaries XDMF file.
    """
    
    # load in the SVM-Tk, GMSH, etc. mesh with meshio.
    meshio_mesh = meshio.read(mio_fpath)
    
    if "tetra" in meshio_mesh.cells_dict.keys():
        xdmf_mesh = create_dolfin_mesh(meshio_mesh, cell_type="tetra")
        
        triangles = {"triangle": meshio_mesh.cells_dict["triangle"]}
        tetra = {"tetra": meshio_mesh.cells_dict["tetra"]}
        subdomains = {"subdomains": [meshio_mesh.cell_data_dict["medit:ref"]["tetra"]]}
        boundaries = {"boundaries": [meshio_mesh.cell_data_dict["medit:ref"]["triangle"]]}
        
        xdmfsubdomain = meshio.Mesh(meshio_mesh.points, tetra, cell_data=subdomains)
        xdmfbndry = meshio.Mesh(meshio_mesh.points, triangles, cell_data=boundaries)
        
    elif "triangle" in meshio_mesh.cells_dict.keys():
        xdmf_mesh = create_dolfin_mesh(meshio_mesh, cell_type="triangle", prune_z=True)
        
        triangles = {"triangle": meshio_mesh.cells_dict["triangle"]}
        lines = {"line": meshio_mesh.cells_dict["line"]}
        subdomains = {"subdomains": [meshio_mesh.cell_data_dict["medit:ref"]["triangle"]]}
        boundaries = {"boundaries": [meshio_mesh.cell_data_dict["medit:ref"]["line"]]}
        
        xdmfsubdomain = meshio.Mesh(meshio_mesh.points, triangles, cell_data=subdomains)
        xdmfbndry = meshio.Mesh(meshio_mesh.points, lines, cell_data=boundaries)
        
    # write the output files.
    meshio.write(xdmf_fpath, xdmf_mesh)
    meshio.write(subdomain_fpath, xdmfsubdomain)
    meshio.write(boundary_fpath, xdmfbndry)


def import_mesh(filename:str):
    """Import a .msh or .mesh file and return a dolfin mesh, domains and facets meshfunctions.
    This function is based upon code from: https://gitlab.enpc.fr/navier-fenics/fenics-optim/-/blob/master/fenics_optim/mesh_utils.py

    Args:
        filename (str): Filename for meshio mesh saved to file.

    Returns:
        mesh (dolfin.Mesh)              : Dolfin mesh object
        domains (dolfin.MeshFunction)   : Dolfin domain meshfunction
        facets (dolfin.Meshfunction)    : Dolfin facets meshfunction
    """
    _, ext = os.path.splitext(filename)
    xdmf_filename = filename.replace(ext, ".xdmf")
    facets_xdmf_filename = filename.replace(ext, "_facets.xdmf")
    
    meshio_mesh = meshio.read(filename)

    if "tetra" in meshio_mesh.cells_dict.keys():
        dim = 3
        domain_mesh = create_dolfin_mesh(meshio_mesh, "tetra")
        facets_mesh = create_dolfin_mesh(meshio_mesh, "triangle")
    elif "triangle" in meshio_mesh.cells_dict.keys():
        dim = 2
        domain_mesh = create_dolfin_mesh(meshio_mesh, "triangle", True)
        facets_mesh = create_dolfin_mesh(meshio_mesh, "line", True)

    meshio.write(xdmf_filename, domain_mesh)
    mesh = dl.Mesh()
    mvc = dl.MeshValueCollection("size_t", mesh, dim)
    with dl.XDMFFile(xdmf_filename) as infile:
        infile.read(mesh)
        infile.read(mvc, "name_to_read")
    domains = dl.MeshFunction("size_t", mesh, mvc)

    meshio.write(facets_xdmf_filename, facets_mesh)
    mvc = dl.MeshValueCollection("size_t", mesh, dim - 1)
    with dl.XDMFFile(facets_xdmf_filename) as infile:
        infile.read(mvc, "name_to_read")
    facets = dl.MeshFunction("size_t", mesh, mvc)

    return mesh, domains, facets


def write_xdmf_to_h5(xdmfmesh: str, subdomainmesh: str, bndrymesh: str, hdf5file: str):
    """Write XDMF mesh data to HDF5 file.

    Args:
        xdmfmesh (str): Path to XDMF mesh file.
        subdomainmesh (str): Path to XDMF mesh subdomains file.
        bndrymesh (str): Path to XDMF mesh boundaries file.
        hdf5file (str): Path to HDF5 file to write to.
    """
    # Read .xdmf mesh into a FEniCS Mesh
    mesh = dl.Mesh()
    with dl.XDMFFile(xdmfmesh) as infile:
        infile.read(mesh)
        
    # Read cell data to a MeshFunction (of dim n)
    n = mesh.topology().dim()
    subdomains = dl.MeshFunction("size_t", mesh, n)
    with dl.XDMFFile(subdomainmesh) as infile:
        infile.read(subdomains, "subdomains")
        
    # Read facet data to a MeshFunction (of dim n-1)
    boundaries = dl.MeshFunction("size_t", mesh, n-1, 0)
    with dl.XDMFFile(bndrymesh) as infile:
        infile.read(boundaries, "boundaries")

    # Write all files into a single h5 file.
    hdf = dl.HDF5File(mesh.mpi_comm(), hdf5file, "w")
    hdf.write(mesh, "/mesh")
    hdf.write(subdomains, "/subdomains")
    hdf.write(boundaries, "/boundaries") 
    hdf.close()


def write_h5_to_xdmf(hdf5file: str, xdmfbase: str):
    mesh = dl.Mesh()
    with dl.HDF5File(mesh.mpi_comm(), hdf5file, "r") as fid:
        # Read mesh data.
        fid.read(mesh, "/mesh", False)
        n = mesh.topology().dim()
        
        # Read subdomain data.
        subdomains = dl.MeshFunction("size_t", mesh, n)
        fid.read(subdomains, "/subdomains")
        
        # Read boundary data.
        boundaries = dl.MeshFunction("size_t", mesh, n-1, 0)
        fid.read(boundaries, "/boundaries")
    
    # Generate XDMF names.
    XDMFMESH = xdmfbase + ".xdmf"
    SUBDOMAINMESH = xdmfbase + "-subdomains.xdmf"
    BNDRYMESH = xdmfbase + "-boundaries.xdmf"

    # Write mesh.
    with dl.XDMFFile(mesh.mpi_comm(), XDMFMESH) as fid:
        fid.write(mesh)

    # Write subdomains.
    with dl.XDMFFile(mesh.mpi_comm(), SUBDOMAINMESH) as fid:
        fid.write(subdomains)
    
    # Write boundaries.
    with dl.XDMFFile(mesh.mpi_comm(), BNDRYMESH) as fid:
        fid.write(boundaries)


def get_mesh_bbox(mesh:dl.Mesh)->np.array:
    """Helper function to get the bounding box of a dolfin mesh.
    WARNING: Only works in serial.

    Args:
        mesh (dl.Mesh): The dolfin mesh object.

    Returns:
        np.array: Array of bounding box coordinates. [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    """
    coords = mesh.coordinates()
    xmin = np.min(coords, axis=0)
    xmax = np.max(coords, axis=0)
    
    return np.vstack((xmin, xmax))


# ----------------------------------------------
# Convenience functions
# ----------------------------------------------
from mpi4py import MPI
def report_mesh_info(mesh: dl.Mesh)->None:
    """Report basic mesh information.
    # NOTE: ghost cells will get double counted by MPI collectives.

    Args:
        mesh (dl.Mesh): The dolfin mesh object.
        
    Returns:
        None: Prints mesh information to stdout.
    """
    SEP = "\n"+"#"*80+"\n"  # stdout separator
    
    # compute mesh information.
    nvertex = MPI.COMM_WORLD.allreduce(mesh.num_vertices(), op=MPI.SUM)
    ncell = MPI.COMM_WORLD.allreduce(mesh.num_cells(), op=MPI.SUM)
    hmax = MPI.COMM_WORLD.allreduce(mesh.hmax(), op=MPI.MAX)
    hmin = MPI.COMM_WORLD.allreduce(mesh.hmin(), op=MPI.MIN)
    vol = dl.assemble(1*dl.dx(mesh))
    
    # report the information.
    if MPI.COMM_WORLD.rank == 0:
        print(SEP, flush=True)
        print("Mesh info:", flush=True)
        print(f"# vertices:\t\t{nvertex}", flush=True)
        print(f"# cells:\t\t{ncell}", flush=True)
        print(f"max cell size (mm):\t{hmax:.2e}", flush=True)
        print(f"min cell size (mm):\t{hmin:.2e}", flush=True)
        if mesh.topology().dim() == 2:
            print(f"Area (mm^2):\t\t{vol:.2e}", flush=True)
        elif mesh.topology().dim() == 3:
            print(f"Volume (mm^3):\t\t{vol:.2e}", flush=True)
        print(SEP, flush=True)


def check_mesh_dimension(mesh: dl.Mesh, zoff: float) -> float:
    """Check the dimension of the mesh and return the z-offset.

    Args:
        mesh (dl.Mesh): The FEniCS mesh object.
        zoff (float): The supplied z-offset argument.

    Returns:
        float: The z-offset. If the mesh is 3D this is a :code:`NoneType`.
    """
    
    # Get the physical dimension of the mesh.
    phys_dim = mesh.topology().dim()
    if (phys_dim == 2) and (mesh.geometry().dim() == 3):
        ZOFF = np.unique(mesh.coordinates()[:, -1])[0]
        print(f"Z-offset set from mesh file: {ZOFF}")
    elif (mesh.geometry().dim() == 2):
        assert zoff is not None, "Cannot read Z-coordinate from mesh. Please provide a value."
        ZOFF = zoff
        print(f"Z-coordinate not determined from mesh.")
        print(f"Using user-provided z-offset:\t{ZOFF}")
    else:
        ZOFF = None  # do not use z-offset if 3D.

    return ZOFF


def load_mesh(comm, mesh_fpath:str) -> None:
    """Load a dolfin mesh from file.

    Args:
        comm (dl.MPI.comm_world): MPI communicator.
        mesh_fpath (str): Path to the mesh file.
    """
    suffix = Path(mesh_fpath).suffix
    
    mesh = dl.Mesh(comm)
    
    if suffix == ".h5":
        with dl.HDF5File(mesh.mpi_comm(), mesh_fpath, "r") as fid:
            fid.read(mesh, "/mesh", False)
    elif suffix == ".xdmf":
        with dl.XDMFFile(mesh.mpi_comm(), mesh_fpath) as fid:
            fid.read(mesh)
    else:
        raise ValueError(f"Unknown mesh file format: {suffix}")

    return mesh


def load_mesh_subs(comm, mesh_fpath:str) -> None:
    """Load a dolfin mesh from file.

    Args:
        comm (dl.MPI.comm_world): MPI communicator.
        mesh_fpath (str): Path to the mesh file.
    """
    suffix = Path(mesh_fpath).suffix
    
    mesh = dl.Mesh(comm)
    
    if suffix == ".h5":
        with dl.HDF5File(mesh.mpi_comm(), mesh_fpath, "r") as fid:
            fid.read(mesh, "/mesh", False)
            
        # for the indicator function.
        n = mesh.topology().dim()
        subs = dl.MeshFunction("size_t", mesh, n)
        bndrys = dl.MeshFunction("size_t", mesh, n-1)

        # attempt to load in subdomains and boundaries.
        try:
            with dl.HDF5File(mesh.mpi_comm(), mesh_fpath, "r") as fid:
                fid.read(subs, "/subdomains")
                fid.read(bndrys, "/boundaries")
        except:
            raise ValueError("Unable to read subdomains or boundaries from provided HDF5 file.")
    else:
        raise ValueError(f"Unknown mesh file format: {suffix}. Was expecting HDF5.")

    return mesh, subs, bndrys

