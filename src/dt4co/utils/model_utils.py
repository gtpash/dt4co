import math
import dolfin as dl
import ufl
import numpy as np

import hippylib as hp

# -------------------------------------
def solveIndicators(mesh:dl.Mesh, subs:dl.MeshFunction, sidx:int)->dl.Function:
    dl.set_log_level(dl.LogLevel.WARNING)  # supress output
    
    Vh_DG0 = dl.FunctionSpace(mesh, "DG", 0)
    chi = dl.Function(Vh_DG0)
    
    # Assemble the dummy variational form to solve for the indicator function.
    chi_test = dl.TestFunction(Vh_DG0)
    dx = dl.Measure("dx", domain=mesh, subdomain_data=subs)
    varf = dl.inner(chi_test, chi)*ufl.dx - ufl.inner(dl.Constant(1.), chi_test)*dx(sidx)
    dl.solve(varf == 0, chi)
    
    return chi


def samplePrior(prior, n: int=1, seed=None)->dl.Vector:
    """Wrapper to sample from a :code:`hIPPYlib` prior.

    Args:
        prior: :code:`hIPPYlib` prior object.
        n: How long to burn in the RNG. Defaults to 1. Useful for drawing multiple samples from the same seed.
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        dl.Vector: sample from prior.
    """
    
    # Get a random normal sample.
    noise = dl.Vector()
    if seed is not None:
        prior.init_vector(noise, "noise")
        for _ in range(n):
            # burn in the smapler.
            rng = hp.Random(seed=seed)
        rng.normal(1.0, noise)
    else:
        for _ in range(n):
            # burn in the sampler.
            prior.init_vector(noise, "noise")
            hp.parRandom.normal(1., noise)
    
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)
    
    return mtrue


def SphericalInitialCondition(dim: int,
                              center: list,
                              r: float,
                              u0: float=0.5,
                              degree: int=1) -> dl.Expression:
    """Spherical (circular) initial condition.

    Args:
        dim (int): Geometric dimension.
        center (list): Coordinates of the sphere or circle center.
        r (float): Radius.
        u0 (float, optional): Initial tumor density. Defaults to 0.5.

    Returns:
        dl.Expression: Initial condition implemented as Dolfin Expression.
    """
    # -------------------------------------
    # NOTE; For an initial condition specified via data (NumPy array) consider: https://fenicsproject.discourse.group/t/interpolate-numpy-ndarray-to-function/6167
    # NOTE: For more complex initial conditions, consider using the UserExpression super class, e.g. https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/cahn-hilliard/demo_cahn-hilliard.py.html?highlight=userexpression
    # -------------------------------------
    
    assert len(center) == dim, f"Geometric dimension does not equal coordinates in 'center'. Dimension should be {dim}."
    
    if dim == 2:
        return dl.Expression("pow(x[0]-cx,2)+pow(x[1]-cy,2) < pow(r,2) ? valin : valout",
                    cx=center[0], cy=center[1], r=r, valin=u0, valout=0., degree=degree)
    elif dim == 3:
        return dl.Expression("pow(x[0]-cx,2)+pow(x[1]-cy,2)+pow(x[2]-cz,2) \
                    < pow(r,2) ? valin : valout",
                    cx=center[0], cy=center[1], cz=center[2], r=r, valin=u0, valout=0, degree=degree)
    else:
        raise ValueError(f"{dim} is not a supported geometric dimension.")
    

def MollifierInitialCondition(dim: int,
                              center: list,
                              r: float,
                              v: float=0.5,
                              degree: int=1) -> dl.Expression:
    
    assert len(center) == dim, f"Geometric dimension does not equal coordinates in 'center'. Dimension should be {dim}."
    
    if dim == 2:
        return dl.Expression("v * a * exp( -(pow(x[0]-cx,2)+pow(x[1]-cy,2) )/ (2*(pow(b,2))) )", 
                        cx=center[0],
                        cy=center[1],
                        a=1/(np.sqrt(2*math.pi)*r),
                        b=r,
                        v=v,
                        degree=degree)
    if dim == 3:
        return dl.Expression("v * a * exp( -( pow(x[0]-cx,2) + pow(x[1]-cy,2) + pow(x[2]-cz,2) ) / (2*(pow(b,2))) )", 
                        cx=center[0],
                        cy=center[1],
                        cz=center[2],
                        a=1/(np.sqrt(2*math.pi)*r),
                        b=r,
                        v=v,
                        degree=degree)
    else:
        raise ValueError(f"{dim} is not a supported geometric dimension.")
    
    
def computeFunctionCenterOfMass(f:dl.Function, Vh:dl.FunctionSpace) -> list:
    """Compute the center of mass of a Dolfin function.
    NOTE: Pass dl.Constant(1.0) to get the center of mass of the domain.

    Args:
        f (dl.Function): The dolfin function.
        Vh (dl.FunctionSpace): Function space of the Dolfin function.

    Returns:
        list: Center of mass coordinates
    """
       
    # interpolate the unit directions 
    xfun = dl.Function(Vh)
    xfun.assign(dl.Expression("x[0]", element=Vh.ufl_element()))
    yfun = dl.Function(Vh)
    yfun.assign(dl.Expression("x[1]", element=Vh.ufl_element()))
    zfun = dl.Function(Vh)
    zfun.assign(dl.Expression("x[2]", element=Vh.ufl_element()))
    
    vol = dl.assemble(f * dl.dx)
    
    xcom = dl.assemble(f * xfun * dl.dx) / vol
    ycom = dl.assemble(f * yfun * dl.dx) / vol
    zcom = dl.assemble(f * zfun * dl.dx) / vol
    
    return [xcom, ycom, zcom]
