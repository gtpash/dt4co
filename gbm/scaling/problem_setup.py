import os
import sys

import dolfin as dl
import ufl
import numpy as np

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
import hippylib as hp

sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
from dt4co.utils.model_utils import MollifierInitialCondition
from dt4co.models import RDTumorVarf

def setupProblem(mesh: dl.Mesh, u0: dl.Function=None) -> tuple:
    """Set up the problem for scaling study.

    Args:
        mesh (dl.Mesh): The mesh object.
        u0 (dl.Function, optional): Initial condition. Defaults to None.

    Returns:
        pde: The problem object.
        mprior: The prior object.
    """
    STATE_DEGREE = 1  # P1 elements for state
    PARAM_DEGREE = 1  # P1 elements for parameter
    DELTA_T = 1.00    # time step [days]
    D0 = 0.03         # reference diffusion coefficient [mm^3/day]
    K0 = 0.03         # reference proliferation rate [1/day]
    RHO_D = 180.0     # correlation length for diffusion
    RHO_K = 180.0     # correlation length for proliferation
    VAR_D = 0.2336    # variance for diffusion
    VAR_K = 0.0682    # variance for proliferation
    
    SEED_VAL = 10     # empirically found to give a good initial tumor seed
    
    t0 = 0.0          # initial time
    tf = 30.0         # final time (3 months)
    
    # Set up the variational spaces for state and parameter.
    Vhu = dl.FunctionSpace(mesh, "Lagrange", STATE_DEGREE)
    Vhmi = dl.FunctionSpace(mesh, "Lagrange", PARAM_DEGREE)
    mixed_element = ufl.MixedElement([Vhmi.ufl_element(), Vhmi.ufl_element()])
    Vhm = dl.FunctionSpace(mesh, mixed_element)  #todo: use vector function space?
    Vh = [Vhu, Vhm, Vhu]
    
    # Set up the variational problem.
    f = dl.Constant(0)
    varf = RDTumorVarf(DELTA_T, f, lumped=False)
    
    bc = []  # homogeneous Neumann for state
    bc0 = []  # homogeneous Neumann for adjoint
    
    # set up the initial condition
    if u0 is None:
        # Assuming the mesh is a box.
        vv = np.max(mesh.coordinates())
        # Place a seed tumor in the middle of the domain.
        u0_expr = MollifierInitialCondition(dim=3, center=[vv/2, vv/2, vv/2], r=vv/16, v=SEED_VAL, degree=2)
        u0 = dl.interpolate(u0_expr, Vhu)
    
    pde = hp.SNES_TimeDependentPDEVariationalProblem(Vh, varf, bc, bc0, u0, t0, tf, is_fwd_linear=False)
    
    # Need to use iterative methods in 3D
    pde.solverA = dl.PETScKrylovSolver('cg', 'petsc_amg')
    pde.solverAadj = dl.PETScKrylovSolver('cg', 'petsc_amg')
    pde.solver_fwd_inc = dl.PETScKrylovSolver('cg', 'petsc_amg')
    pde.solver_adj_inc = dl.PETScKrylovSolver('cg', 'petsc_amg')
    
    # Crank down the tolerance of the linear solvers.
    pde.solverA.parameters["relative_tolerance"] = 1e-12
    pde.solverA.parameters["absolute_tolerance"] = 1e-20
    pde.solverAadj.parameters = pde.solverA.parameters
    pde.solver_fwd_inc.parameters = pde.solverA.parameters
    pde.solver_adj_inc.parameters = pde.solverA.parameters
    
    # Set the prior mean.
    mu = dl.Function(Vh[hp.PARAMETER])
    mu.assign(dl.Constant([np.log(D0), np.log(K0)]))
    
    # Get the prior coefficients, and set up the prior.
    diff_coeffs = hp.BiLaplacianComputeCoefficients(VAR_D, RHO_D, ndim=3)
    prolif_coeffs = hp.BiLaplacianComputeCoefficients(VAR_K, RHO_K, ndim=3)
    mprior = hp.VectorBiLaplacianPrior(Vh[hp.PARAMETER], [diff_coeffs[0], prolif_coeffs[0]], [diff_coeffs[1], prolif_coeffs[1]], mean=mu.vector(), robin_bc=True)

    return pde, mprior


def setupAdjoint(pde, x):
    """Set up the adjoint problem for scaling study.

    Args:
        pde: The PDE object.
        x: The list of solution vectors.

    Returns:
        misfit: The misfit object.
    """
    REL_NOISE = 0.01
    max_state = x[hp.STATE].norm("linf", "linf")
    noise_std_dev = REL_NOISE * max_state
    
    # Set up misfit object.
    misfits = []
    for t in pde.times:
        misfit_t = hp.ContinuousStateObservation(pde.Vh[hp.STATE], ufl.dx, pde.adj_bc)
        misfit_t.d.axpy(1., x[hp.STATE].view(t))
        hp.parRandom.normal_perturb(noise_std_dev, misfit_t.d)
        misfit_t.noise_variance = noise_std_dev*noise_std_dev
        misfits.append(misfit_t)
    
    misfit = hp.MisfitTD(misfits, pde.times)
    
    return misfit
