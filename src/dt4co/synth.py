import numpy as np
import dolfin as dl
import ufl
import nibabel

import hippylib as hp

from .experiments import Experiment
from .models import radioModel, chemoModel, mollifiedPWRDTXTumorVarf, mollifiedPWRDTumorVarf
from .models import mollifiedRDTumorVarf, mollifiedRDTXTumorVarf
from .utils.data_utils import niftiPointwiseObservationOp
from .utils.model_utils import samplePrior
from .utils.parallel import numpy2Vec

def build_rt_timeline(start=7., num_weeks=6):
    tx_start = start
    tx_week = np.array([0, 1, 2, 3, 4])
    tx_days = tx_week
    for i in range(1, num_weeks):
        tx_days = np.concatenate((tx_days, i*7 + tx_week))
    
    tx_days = tx_days + tx_start
    return tx_days


def build_ct_timeline(start=7., num_weeks=6):
    tx_start = start
    tx_week = np.array([0, 1, 2, 3, 4, 5, 6])
    tx_days = tx_week
    for i in range(1, num_weeks):
        tx_days = np.concatenate((tx_days, i*7 + tx_week))
    
    tx_days = tx_days + tx_start
    return tx_days


def setup_stupp(tx_start=14.0):
    rt_days = build_rt_timeline(start=tx_start, num_weeks=6)
    ct_days = build_ct_timeline(start=tx_start, num_weeks=6)
    return rt_days, ct_days


def setup_data_collection_timeline(last_day, pred_date, step=4.0):
    start = 0.0
    num = np.rint(last_day / step)
    
    img_days = start + np.arange(0, num) * step
    
    # add the prediction date to the image days
    img_days = np.append(img_days, last_day + pred_date)
    
    return img_days

class synthExperiment(Experiment):
    """Class to hold parameters and setup the synthetic experiment.
    """
    
    def __init__(self):
        # Simulation parameters.
        self.STATE_DEGREE = 1        # degree of the state finite element space
        self.PARAM_DEGREE = 1        # degree of the parameter finite element space
        self.DELTA_T = 1.0           # time step [day]
        self.DG_TRUE = 0.03          # diffusion coefficient (for gray matter) [mm^d / day]
        self.DW_TRUE = 0.30          # diffusion coefficient (for white matter) [mm^d / day]
        self.K_TRUE = 0.15           # proliferation rate coefficient [1/day] (excluding radiotherapy effect)
        self.NOISE = 0.02            # noise level for the data
        
        # for the BIP prior.
        self.D0 = 0.05               # initial guess for diffusion coefficient [mm^2 / day]
        # self.VAR_D = 0.2336          # variance of the log-diffusion coefficient
        self.VAR_D = 0.05
        self.RHO_D = 180.0           # diffusion correlation length [mm]
        self.K0 = 0.10               # initial guess for proliferation rate coefficient [1/day] (excluding radiotherapy effect)
        # self.VAR_K = 0.0682          # variance of the log-proliferation rate
        self.VAR_K = 0.02
        self.RHO_K = 180.0           # proliferation rate correlation length [mm]
        
        self.RHO_DG = 180.0          # diffusion correlation length in gray matter [mm]
        self.RHO_DW = 180.0          # diffusion correlation length in white matter [mm]
        self.VAR_DW = 0.5           # variance of the log-diffusion coefficient (for white matter)
        self.VAR_DG = 0.5           # variance of the log-diffusion coefficient (for gray matter)
        
        # Define the radiotherapy term.
        self.RT_DOSE = 2.0              # radiotherapy dose [Gy]
        self.ALPHA = 0.025              # LQ model alpha radiosensitivity parameter [1/day]
        self.alpha_beta_ratio = 10.0
        self.m_width = None             # do not use the mollified version

        # Define the chemotherapy term.
        self.CT_EFFECT = 0.9      # chemotherapy surviving fraction
        self.BETA = 24. / 1.8      # clearance rate of the chemotherapy [1/day]

        # Define the boundary conditions.
        self.bc = []   # homogeneous Neumann for state
        self.bc0 = []  # homogeneous Neumann for adjoint
        
    def setupFunctionSpaces(self, mesh, mle: bool=False):
        Vhu = dl.FunctionSpace(mesh, "Lagrange", self.STATE_DEGREE)
        
        if mle:
            Vhmi = dl.FunctionSpace(mesh, "Real", 0)  # use a constant function space for the parameter
        else:
            Vhmi = dl.FunctionSpace(mesh, "Lagrange", self.PARAM_DEGREE)
                
        # P1 x P1 x P1 for m_dg, m_dw, m_k
        mixed_element = ufl.MixedElement([Vhmi.ufl_element(), Vhmi.ufl_element(), Vhmi.ufl_element()])
        Vhm = dl.FunctionSpace(mesh, mixed_element)
        Vh = [Vhu, Vhm, Vhu]

        return Vh
    
    def setupTXModels(self, tx_start:int=14.0):
        """Set up the radiotherapy and chemotherapy models.
        """
        rt_days, ct_days = setup_stupp(tx_start=tx_start)
        rt_doses = self.RT_DOSE * np.ones_like(rt_days)
        
        radio_model = radioModel(tx_days=rt_days, tx_doses=rt_doses, alpha=self.ALPHA, alpha_beta_ratio=self.alpha_beta_ratio)
        chemo_model = chemoModel(ct_days, ct_effect=self.CT_EFFECT, beta=self.BETA)
        
        return radio_model, chemo_model
    
    def setupVariationalProblem(self, Vh, u0, t0, tf, chi_gm: dl.Function, exptype:str, radio_model, chemo_model, sparam=None, moll: bool=False):
        """NOTE: mollified by default.
        """
        
        if exptype == "rd":
            # reaction-diffusion model
            varf = mollifiedPWRDTumorVarf(self.DELTA_T, chi_gm=chi_gm)
        elif exptype == "rdtx":
            # reaction-diffusion model with radiotherapy + chemotherapy
            varf = mollifiedPWRDTXTumorVarf(self.DELTA_T, rtmodel=radio_model, ctmodel=chemo_model, chi_gm=chi_gm)
        else:
            raise ValueError("Unknown experiment type.")
        
        pde = hp.SNES_TimeDependentPDEVariationalProblem(Vh, varf, self.bc, self.bc0, u0, t0, tf, is_fwd_linear=False, solver_params=sparam)
        
        # Use iterative linear solvers.
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
        
        return pde
    
    def trueParameter(self, Vh: list, sample: bool=False) -> dl.Vector:
        """Return the prior mean for the parameters.

        Args:
            Vh (list): The function space list.
            
        Returns:
            dl.Vector: The prior mean function.
        """
        
        # assign the true parameter mean (m_dg, m_dw, m_k).
        mu = dl.Function(Vh[hp.PARAMETER])
        mu.assign(dl.Constant([np.log(self.DG_TRUE), np.log(self.DW_TRUE), np.log(self.K_TRUE)]))
        
        if sample:
            # BiLaplacian prior for the parameters.
            phys_dim = Vh[hp.STATE].mesh().topology().dim()  # get the physical dimension of the mesh
            
            diff_coeffs = hp.BiLaplacianComputeCoefficients(self.VAR_D, self.RHO_D, ndim=phys_dim)
            prolif_coeffs = hp.BiLaplacianComputeCoefficients(self.VAR_K, self.RHO_K, ndim=phys_dim)
            mprior = hp.VectorBiLaplacianPrior(Vh[hp.PARAMETER], [diff_coeffs[0], diff_coeffs[0], prolif_coeffs[0]], [diff_coeffs[1], diff_coeffs[1], prolif_coeffs[1]], mean=mu.vector(), robin_bc=True)
            
            mtrue = samplePrior(mprior, n=1, seed=42)
            return mtrue
        else:
            return mu.vector()
        
    
    def setupPrior(self, Vh, mle: bool=False):
        """Set up the prior for the parameters.
        Only for use with the RD / RD + TX models (2 parameters)
        """

        # Set up the prior mean (md0, mk0)
        mu = dl.Function(Vh[hp.PARAMETER])
        mu.assign(dl.Constant([np.log(self.D0), np.log(self.K0)]))

        if mle:
            # maximum likelihood estimation, use a dummy prior.
            dummy_cov = np.eye(2)  # no preconditioning for the Newton solve
            mprior = hp.GaussianRealPrior(Vh[hp.PARAMETER], covariance=dummy_cov)
            mprior.R.zero()  # zero out the regularization
        else:
            # BiLaplacian prior for the parameters.
            phys_dim = Vh[hp.STATE].mesh().topology().dim()  # get the physical dimension of the mesh
            
            diff_coeffs = hp.BiLaplacianComputeCoefficients(self.VAR_D, self.RHO_D, ndim=phys_dim)
            prolif_coeffs = hp.BiLaplacianComputeCoefficients(self.VAR_K, self.RHO_K, ndim=phys_dim)
            mprior = hp.VectorBiLaplacianPrior(Vh[hp.PARAMETER], [diff_coeffs[0], prolif_coeffs[0]], [diff_coeffs[1], prolif_coeffs[1]], mean=mu.vector(), robin_bc=True)
        
        return mprior
    
    def setupBIPFunctionSpaces(self, mesh, mle: bool=False):
        Vhu = dl.FunctionSpace(mesh, "Lagrange", self.STATE_DEGREE)
        
        if mle:
            Vhmi = dl.FunctionSpace(mesh, "Real", 0)  # use a constant function space for the parameter
        else:
            Vhmi = dl.FunctionSpace(mesh, "Lagrange", self.PARAM_DEGREE)
                
        # P1 x P1 x P1 for m_dg, m_dw, m_k
        mixed_element = ufl.MixedElement([Vhmi.ufl_element(), Vhmi.ufl_element()])
        Vhm = dl.FunctionSpace(mesh, mixed_element)
        Vh = [Vhu, Vhm, Vhu]

        return Vh
    
    def setupBIPVariationalProblem(self, Vh, u0, t0, tf, exptype, sparams, radio_model=None, chemo_model=None):
        """Convencience function to set up the BIP variational problem.
        
        Similar functionality to the `ExperimentFactory` class.
        """
        
        if exptype == "rd":
            # reaction-diffusion model
            varf = mollifiedRDTumorVarf(self.DELTA_T)
        elif exptype == "rdtx":
            # reaction-diffusion model with radiotherapy + chemotherapy
            varf = mollifiedRDTXTumorVarf(self.DELTA_T, rtmodel=radio_model, ctmodel=chemo_model)
        else:
            raise ValueError("Unknown experiment type.")
        
        pde = hp.SNES_TimeDependentPDEVariationalProblem(Vh, varf, self.bc, self.bc0, u0, t0, tf, is_fwd_linear=False, solver_params=sparams)
        
        # Use iterative linear solvers.
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
        
        return pde
    
    def spoofMisfitTD(self, visits: list, visit_days, Vh:dl.FunctionSpace, noise_var: float, exnii: str=None):
        """Build a misfit object for the synthetic experiment.

        Args:
            visits (list): List of file paths to the tumor images to be used to compute the misfit.
            visit_days (array-like): Array of the visit days.
            Vh (dl.FunctionSpace): The (state) function space.
            noise_var (float): The noise variance.
            exnii (str): The example NIfTI image file.
            pointwise (bool, optional): Whether or not to use discrete observations. Defaults to True.

        Returns:
            MisfitTD: The time-dependent misfit object.
        """
        MISFITS = []
    
        assert exnii is not None, "Need an example NIfTI image to build the observation operator."
        
        # grab example image to assemble the observation operator (first tumor image)
        obsOp = niftiPointwiseObservationOp(exnii, Vh)
        
        for i, visit_tumor in enumerate(visits):
            
            # load in the NIfTI image data and flatten (every process will have a copy)
            npdata = nibabel.load(visit_tumor).get_fdata().flatten()
            
            # initialize a PETScVector to store the data (compatible with the observation operator)
            dvec = dl.PETScVector(obsOp.mpi_comm())
            obsOp.init_vector(dvec, 0)
            
            numpy2Vec(dvec, npdata)
            
            misfit = hp.DiscreteStateObservation(obsOp, dvec.copy(), noise_variance=noise_var)
            MISFITS.append(misfit)
        
        misfit_obj = hp.MisfitTD(MISFITS, visit_days)
        
        return misfit_obj
    
    def spoofContinuousMisfitTD(self, uh, visit_days, Vh:dl.FunctionSpace, noise_var:float) -> hp.MisfitTD:
        """Build a misfit object for the synthetic experiment.
        Apply noise to the state observations at the visit days.

        Args:
            uh (hp.TimeDependentVector): The time-dependent state vector.
            visit_days (array-like): Array of the visit days.
            Vh (dl.FunctionSpace): The (state) function space.
            noise_var (float): The noise variance.

        Returns:
            MisfitTD: The time-dependent misfit object.
        """
        MISFITS = []
        
        for i, visit_day in enumerate(visit_days):
            uhelp = dl.Function(Vh)
            uhelp.vector().zero()
            uhelp.vector().axpy(1., uh.view(visit_day))
            
            misfit = hp.ContinuousStateObservation(Vh, dl.dx, self.bc, uhelp.vector().copy(), noise_variance=noise_var)
            MISFITS.append(misfit)
        
        misfit_obj = hp.MisfitTD(MISFITS, visit_days)
        
        return misfit_obj