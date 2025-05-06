import dolfin as dl
import hippylib as hp
import h5py

def write_mv_to_h5(comm, data: list | hp.MultiVector, Vh, filename:str, name=None) -> None:
    """Write a list of functions to an HDF5 file.

    Args:
        comm: MPI communicator.
        data (list | hp.MultiVector): List of dolfin Vectors or hIPPYlib MutliVector to be exported.
        Vh (dl.FunctionSpace): FEniCS function space to project vectors onto.
        filename (str): Filename to which the functions will be written.
        name (optional): List of dataset names. Alternatively a string to be incremented. Defaults to None. If None, the function names will be "000000", "000001", etc. (default integer width: 6 characters). By default, data is stored in the "/data" group.
    """
    
    # check data type and get number of functions to write.
    assert isinstance(data, list) or isinstance(data, hp.MultiVector), "Data must be a list of dolfin Vectors or a hIPPYlib MultiVector."
    if isinstance(data, hp.MultiVector):
        ndata = data.nvec()
    else:
        ndata = len(data)
    
    DEFAULT_GROUP = "/data"
    if name is None:
        # build list of names if none is provided.
        name = [f"{DEFAULT_GROUP}/{i:06d}" for i in range(len(data))]
    elif isinstance(name, str):
        # assumed to be a single name, but a series of functions.
        NAME_2_READ = name
        name = [f"{DEFAULT_GROUP}/{NAME_2_READ}/{i:06d}" for i in range(len(data))]
    else:
        # else, use the provided names.
        assert len(name) == ndata, "Number of names must match number of functions."
    
    with dl.HDF5File(comm, filename, "w") as fid:
        fid.write(Vh.mesh(), "/mesh")  # need to write the mesh for parallel read back.
        
        # iterate through and write each function to file.
        for i in range(ndata):
            fid.write(hp.vector2Function(data[i], Vh, name=name[i]), name[i])


def read_mv_from_h5(comm, mv: hp.MultiVector, Vh, filename:str, name=None) -> None:
    """Read a list of functions from an HDF5 file.

    Args:
        comm: MPI communicator.
        mv (hp.MultiVector): hIPPYlib MultiVector to store the read functions.
        Vh (dl.FunctionSpace): FEniCS function space to project vectors onto.
        filename (str): Filename from which the functions will be read.
        name (optional): List of function names. Alternatively a string to be incremented. Defaults to None. If None, the function names will be "000000", "000001", etc. (default integer width: 6 characters). By default, data is stored in the "/data" group.
    """
    
    # check data type and get number of functions to read.
    assert isinstance(mv, hp.MultiVector), "Must supply hIPPYlib MultiVector to store data."
    
    mv.zero()               # zero out the MultiVector.
    fun = dl.Function(Vh)   # temporary function to read into.
    ndata = mv.nvec()       # number of functions to read.
    
    DEFAULT_GROUP = "/data"
    if name is None:
        # build list of names if none is provided.
        names = [f"{DEFAULT_GROUP}/{i:06d}" for i in range(ndata)]
    elif isinstance(name, str):
        # assumed to be a single name, but a series of functions.
        NAME_2_READ = name
        name = [f"{DEFAULT_GROUP}/{NAME_2_READ}/{i:06d}" for i in range(ndata)]
    else:
        # else, use the provided names.
        assert len(name) == ndata, "Number of names must match number of functions."
    
    with dl.HDF5File(comm, filename, "r") as fid:
    # iterate through and read each function from file.
        for i in range(mv.nvec()):
            fid.read(fun, name[i])  # read the function.
            mv[i].axpy(1., fun.vector())  # copy the function data to the MultiVector.


def write_mv_to_xdmf(comm, data: list | hp.MultiVector, Vh, filename:str, name=None) -> None:
    """Write a list of functions to an HDF5 file.

    Args:
        comm: MPI communicator.
        data (list or hp.MultiVector): List of dolfin Vectors or a hIPPYlib MultiVector to be exported.
        Vh (dl.FunctionSpace): FEniCS function space to project vectors onto.
        filename (str): Filename to which the functions will be written.
        name (optional): Name for data series. Defaults to None. If None, the series name will be "data".
    """
    
    # check data type and get number of functions to write.
    assert isinstance(data, list) or isinstance(data, hp.MultiVector), "Data must be a list of dolfin Vectors or a hIPPYlib MultiVector."
    if isinstance(data, hp.MultiVector):
        ndata = data.nvec()
    else:
        ndata = len(data)
    
    if name is None:
        # build list of names if none is provided.
        NAME_2_READ = "data"
    else:
        # assumed to be a single name, but a series of functions.
        assert isinstance(name, str), "Please provide the name as a string."
        NAME_2_READ = name

    with dl.XDMFFile(comm, filename) as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
        for i in range(ndata):
            fid.write(hp.vector2Function(data[i], Vh, name=NAME_2_READ), i)


def getGroupSize(file:str, name:str) -> int:
    """Get size of a Group in an HDF5 file.
    Default group name is "/data" and the supplied name is the dataset name.

    Args:
        file (str): Path to the HDF5 file.
        name (str): Name of the dataset.

    Returns:
        int: The size of the dataset.
    """
    with h5py.File(file, "r") as fid:
        n = len(fid['data'][name])
        
    return n
