from petsc4py import PETSc
import numpy as np

def root_print(comm, *args, **kwargs) -> None:
    if comm.rank == 0:
        print(*args, **kwargs, flush=True)


def gather_to_zero(pvec):
    """
    Gather the global PETSc vector on rank zero.
    """
    g20, pvec_full = PETSc.Scatter().toZero(pvec)
    g20.scatter(pvec, pvec_full, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    # g20.scatter(pvec, pvec_full, False, PETSc.ScatterMode.FORWARD)

    g20.destroy()  # deallocate the scatter context
    
    return pvec_full


def scatter_from_zero(pvec0):
    """
    Scatter the global PETSc vector from rank zero.
    """
    g20, pvec = PETSc.Scatter().toAll(pvec0)
    g20.scatter(pvec0, pvec, False, PETSc.ScatterMode.FORWARD)

    g20.destroy()  # deallocate the scatter context

    return pvec


def numpy2Vec(vec, np_arr):
    """Write numpy array to distributed PETSc vector.

    Args:
        vec (dl.PETScVector): The PETSc vector to write to.
        np_arr (np.ndarray): The numpy array to write to the PETSc vector.
        
    Returns:
        None: The PETSc vector is modified in place.
    """
    
    # get the local to global map and insert the data.
    loc_to_glob =  vec.vec().getLGMap()
    procLocalIndices = loc_to_glob.getIndices()
    localrange = np.arange(procLocalIndices.shape[0]).astype('int32')  # PETSc defaults to 32-bit integers.
    vec.vec().setValuesLocal(localrange, np_arr[procLocalIndices], PETSc.InsertMode.INSERT_VALUES)
    vec.apply("")  # set the object into the right state.
