import dolfin as dl
from dolfin import *  # see BUG

def svmtk_refine(mesh: dl.Mesh, subdomains: dl.MeshFunction, boundaries: dl.MeshFunction, outfname: str, markers: dl.MeshFunction=None):
    """For refinement of specific region of mesh (potentially tumor location?)
    See Section 4.5.2 of Rognes et. al.

    Args:
        mesh (dl.Mesh): Dolfin mesh to be refined.
        subdomains (dl.MeshFunction): Original subdomains.
        boundaries (dl.MeshFunction): Original boundaries.
        markers (dl.MeshFunction): Boolean marker for where to refine (where the tumor is).
        outfname (str): Filename to save refined mesh.

    """ 
    # Bind and compile the C++ code for dolfin adapt.
    cpp_binding = """
    #include<pybind11/pybind11.h>
    #include<dolfin/adaptivity/adapt.h>
    #include<dolfin/mesh/Mesh.h>
    #include<dolfin/mesh/MeshFunction.h>
    
    namespace py = pybind11;
    
    PYBIND11_MODULE(SIGNATURE, m) {
        m.def("adapt", (std::shared_ptr<dolfin::MeshFunction<std::size_t>> (*)(const dolfin::MeshFunction<std::size_t>&, std::shared_ptr<const dolfin::Mesh>)) &dolfin::adapt, py::arg("mesh_function"), py::arg("adapted_mesh"));
        m.def("adapt", (std::shared_ptr<dolfin::Mesh> (*)(const dolfin::Mesh&)) &dolfin::adapt );
        m.def("adapt", (std::shared_ptr<dolfin::Mesh> (*)(const dolfin::Mesh&,const dolfin::MeshFunction<bool>&)) &dolfin::adapt );
    }
    """
    adapt = dl.compile_cpp_code(cpp_binding).adapt
    
    # Initialize connections between all mesh entitites
    # use a refinement algorithm that remembers parent facets
    mesh.init()
    
    # KNOWN BUG: https://fenicsproject.org/qa/6719/using-adapt-on-a-meshfunction-looking-for-a-working-example/
    parameters["refinement_algorithm"] = "plaza_with_parent_facets"
    
    
    # Get markers from the NIfTI file containing tumor segmentation data.
    
    if not markers:
        # Refine mesh according to the markers.
        new_mesh = adapt(mesh)
        
        # Update subdomain and boundary markers.
        adapted_subdomains = adapt(subdomains, new_mesh)
        adapted_boundaries = adapt(boundaries, new_mesh)
    else:
        raise NotImplementedError("Haven't implemented refinement for markers")
    
    print(f"Original mesh had {mesh.num_cells()} cells.")
    print(f"Refined mesh has {new_mesh.num_cells()} cells.")
    
    # Write the adapted mesh to file(s).
    hdf = dl.HDF5File(new_mesh.mpi_comm(), outfname, "w")
    hdf.write(new_mesh, "/mesh")
    hdf.write(adapted_subdomains, "/subdomains")
    hdf.write(adapted_boundaries, "/boundaries")
    hdf.close()