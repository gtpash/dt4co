# ----------------------------------------------
# SVM-Tk Functionality
# These scripts are (slight modifications of those) from the book:
# Mathematical Modeling of the Human Brain From Magnetic Resonance Images to Finite Element Simulation
# Kent-Andre Mardal
# Marie E. Rognes
# Travis B. Thompson
# Lars Magnus Valnes
# ----------------------------------------------
import SVMTK as svmtk

# ----------------------------------------------
# General SVM-Tk helpers
# ----------------------------------------------

def svmtk_slice_surfs(plane: tuple, surfs: list[svmtk.Surface], resolution: int, smap: dict=None,  tags: dict=None) -> svmtk.Slice:
    """Generate a slice of surfaces and mesh it.
    
    NOTE: There are known mesh quality issues.
            see SVMTK issue: https://github.com/SVMTK/SVMTK/issues/31

    Args:
        plane (tuple): The plane to slice the surfaces.
        surfs (list[svmtk.Surface]): List of SVMTK Surfaces.
        resolution (int): Mesh resolution for the slice.
        smap (dict): The subdomain map.
        tags (dict): Dictionary of tags for each subdomain.

    Returns:
        svmtk.Slice: The Slice object.
    """
        
    # Slice the surface.
    slce = svmtk.Slice(*plane)
    slce.slice_surfaces(surfs)
    slce.create_mesh(resolution)
    
    # Add subdomains to the slice (if applicable).
    if smap is not None:
        assert tags is not None, "If smap is provided, tags must be provided."
        slce.add_surface_domains(surfs, smap)
    else:
        slce.add_surface_domains(surfs)
    
    surf = slce.export_as_surface()
    
    # Digging in the bag of tricks to try and get a good mesh...
    surf.repair_self_intersections()            # seems to help
    surf.isotropic_remeshing(1.0, 5, False)     # attempt to remove degenerate elements
    surf.separate_narrow_gaps()                 # seems to help
    
    slce.keep_largest_connected_component()     # get rid of "islands"
    
    return slce


# todo: deprecate
def svmtk_create_slice_mesh(stlfile: str, output: str, plane: tuple, resolution: float=32):
    """Slice a surface by a plane and mesh it.
    
    :code:`WARNING`: This is quite brittle. The sliced meshes are delicate.

    Args:
        stlfile (str): STL data file.
        output (str): Filename of where to save volume mesh.
        resolution (float, optional): Resolution of meshed object. Defaults to 32.
        plane (tuple): Plane to slice mesh.
    """
    # Load input file.
    surf = svmtk.Surface(stlfile)
    
    # Slice the surface.
    slce = svmtk.Slice(*plane)
    slce.slice_surfaces([surf])
    slce.create_mesh(resolution)
    slce.add_surface_domains([surf])
    surf2 = slce.export_as_surface()
    
    # Digging in the bag of tricks to try and get a good mesh...
    surf2.isotropic_remeshing(1.0, 3, False)    # attempt to remove degenerate elements
    surf2.separate_close_vertices()             # seems to help
    
    # try to repair self intersections.
    try:
        surf2.repair_self_intersections()           # seems to help
    except:
        pass

    # Write the mesh to the output file.
    surf2.save(output)
    

def svmtk_create_volume_mesh(stlfile: str, output: str, resolution: float=16):
    """Create volume mesh using SVMTK.

    Args:
        stlfile (str): STL data file.
        output (str): Filename of where to save volume mesh.
        resolution (float, optional): Resolution of meshed object. Defaults to 16.
    """
    # Load input file.
    surface = svmtk.Surface(stlfile)
    
    # Generate the volume mesh
    domain = svmtk.Domain(surface)
    domain.create_mesh(resolution)
    
    # Write the mesh to the output file.
    domain.save(output)


def svmtk_remesh_surface(stlinput: str, stloutput: str, L: float=1.0, n: int=3, do_not_move_boundary_edges: bool=False):
    """Remesh surface with SVMTK.
    Based on code from mri2fem (https://github.com/kent-and/mri2fem/blob/master/mri2fem/mri2fem/chp3/remesh_surface.py)

    Args:
        stlinput (str): Input STL file.
        stloutput (str): Output STL file.
        L (float): Maximum edge length of edge cell.
        n (int): Higher value yields finer mesh.
        do_not_move_boundary_edges (bool, optional): Whether or not SVMTK can move boundary. Defaults to False.
    """
    # Load input STL file.
    surface = svmtk.Surface(stlinput)
    
    # Remesh surface.
    surface.isotropic_remeshing(L, n, do_not_move_boundary_edges)
    
    # Save remeshed STL surface.
    surface.save(stloutput)


def svmtk_smooth_surface(stlinput: str, stloutput: str, n: int=1, eps: float=1.0, preserve_volume: bool=True):
    """Smooth surface with SVMTK.
    Based on code from mri2fem (https://github.com/kent-and/mri2fem/blob/master/mri2fem/mri2fem/chp3/smooth_surface.py)

    Args:
        stlinput (str): Input STL file.
        stloutput (str): Output STL file.
        n (int, optional): Number of times to run process. Defaults to 1.
        eps (float, optional): Strength of smoothing. Defaults to 1.0. Range 0.0 (low) to 1.0 (high).
        preserve_volume (bool, optional): Whether or not to preserve volume. Defaults to True.
    """
    # Load input STL file.
    surface = svmtk.Surface(stlinput)
    
    # Smooth using Taubin smoothing if volume is preserved, Laplacian otherwise.
    if preserve_volume:
        surface.smooth_taubin(n)
    else:
        surface.smooth_laplacian(eps, n)

    # Save smoothed STL file.
    surface.save(stloutput)


# ----------------------------------------------
# Brain domain specific SVM-Tk helpers
# ----------------------------------------------

def svmtk_create_gw_mesh(pialstl: str, whitestl: str, output: str, resolution: int=32):
    """Create a tagged mesh with gray and white matter

    Args:
        pialstl (str): STL file containing pial information.
        whitestl (str): STL file containing white matter information.
        output (str): .mesh file to save the mesh.
        resolution (int, optional): Resolution of the mesh. Defaults to 32.
    """
    # Load the surfaces into SVM-Tk and combine in list
    pial = svmtk.Surface(pialstl)
    white = svmtk.Surface(whitestl)
    surfaces = [pial, white]
    
    # Create a map for the subdomains with tags
    # 1 for inside the first and outside the second ("10")
    # 2 for inside the first and inside the second ("11")
    smap = svmtk.SubdomainMap()
    smap.add("10", 1)
    smap.add("11", 2)
    
    # Create a tagged domain from the list of surfaces and the map
    domain = svmtk.Domain(surfaces, smap)
    
    # Create and save the volume mesh
    domain.create_mesh(resolution)
    domain.save(output)


def svmtk_create_gwv_mesh(pialstl: str, whitestl: str, ventstl: str, output: str, resolution: int=32, remove_ventricles: bool=True, plane: tuple=None):
    """Create mesh with gray/white/ventricle segmentations.
    Primarily to be used for meshing a single hemisphere.

    Args:
        pialstl (str): Path to pial STL.
        whitestl (str): Path to white matter STL.
        ventstl (str): Path to ventricle STL.
        output (str): Path for saving output mesh.
        resolution (int, optional): Mesh resolution. Defaults to 32.
        remove_ventricles (bool, optional): Whether or not to remove ventricles. Defaults to True.
        plane (tuple, optional): Plane to slice mesh. Defaults to None. (2D mesh if not None)
    """
    # Create surfaces from STL files.
    pial = svmtk.Surface(pialstl)
    white = svmtk.Surface(whitestl)
    ventricles = svmtk.Surface(ventstl)
    surfaces = [pial, white, ventricles]
    
    # Define identifying tags for different regions.
    tags = {"pial": 1, "white": 2, "ventricle": 3}
    
    # Define the corresponding subdomain map.
    smap = svmtk.SubdomainMap()
    smap.add("100", tags["pial"])
    smap.add("110", tags["white"])
    smap.add("111", tags["ventricle"])
    
    # Mesh and tag domain from the surfaces and the map.
    if plane is not None:        
        # 2D mesh, create slice.
        mesh = svmtk_slice_surfs(plane, surfaces, resolution, smap, tags)
    else:
        # 3D mesh, create domain.
        mesh = svmtk.Domain(surfaces, smap)
        mesh.create_mesh(resolution)
    
    # Remove ventricle subdomain.
    if remove_ventricles:
        mesh.remove_subdomain(tags["ventricle"])

    # Save the mesh.
    mesh.save(output)


def svmtk_fullbrain_five_domain(stls: list, output: str, resolution: int=32, remove_ventricles: bool=True, plane: tuple=None):
    """Create full brain mesh from left/right pial/white matter and ventricles.

    Args:
        stls (list): Paths to: [left pial, right pial, left white, right white, ventricles]
        output (str): File to save resultant mesh to.
        resolution (int, optional): Resolution of resultant mesh. Defaults to 32.
        remove_ventricles (bool, optional): Whether or not to remove the ventricles. Defaults to True.
        plane (tuple, optional): Plane to slice mesh. Defaults to None. (2D mesh if not None)
    """
    # Load each of the Surfaces
    surfaces = [svmtk.Surface(stl) for stl in stls]
    
    # Take the union of the left (#3) and right (#4) white surface
    # put the result into the (former left) white surface
    surfaces[2].union(surfaces[3])
    
    # Drop the right white surface from the list.
    surfaces.pop(3)
    
    # Define identifying tags for the different regions.
    tags = {"pial": 1, "white": 2, "ventricle": 3}
    
    # Label the different regions.
    smap = svmtk.SubdomainMap()
    smap.add("1000", tags["pial"])
    smap.add("0100", tags["pial"])
    smap.add("1010", tags["white"])
    smap.add("0110", tags["white"])
    smap.add("1110", tags["white"])
    smap.add("1011", tags["ventricle"])
    smap.add("0111", tags["ventricle"])
    smap.add("1111", tags["ventricle"])
    
    # Mesh and tag domain from the surfaces and the map.
    if plane is not None:
        # 2D mesh, create slice.
        mesh = svmtk_slice_surfs(plane, surfaces, resolution, smap, tags)
    else:
        # 3D mesh, create domain.
        mesh = svmtk.Domain(surfaces, smap)
        mesh.create_mesh(resolution)
        
    # Remove ventricle subdomain.
    if remove_ventricles:
        mesh.remove_subdomain(tags["ventricle"])
    
    # Save mesh.
    mesh.save(output)


def svmtk_dti():
    """This will require DTI data which has not been used before.
    """
    raise NotImplementedError


def svmtk_combine_hemispheres(rpialstl: str, lpialstl: str, rwhitestl: str, lwhitestl: str):
    """Repair overlapping surfaces as in 4.3.1 of Rognes et. al.

    Args:
        rpialstl (str): Path to right pial STL file.
        lpialstl (str): Path to left pial STL file.
        rwhitestl (str): Path to right white matter STL file.
        lwhitestl (str): Path to left white matter STL file.
    """
    # Input Surfaces
    rpial = svmtk.Surface(rpialstl)
    lpial = svmtk.Surface(lpialstl)
    rwhite = svmtk.Surface(rwhitestl)
    lwhite = svmtk.Surface(lwhitestl)
    
    # Create white matter surface as union of hemispheres.
    white = svmtk.union_partially_overlapping_surfaces(rwhite, lwhite)
    
    # Separate overlapping and close vertices between the left/right pial surfaces
    # but only outside the optional third argument (white surface in this case)
    svmtk.separate_overlapping_surfaces(rpial, lpial, white)
    svmtk.separate_close_surfaces(rpial, lpial, white)
