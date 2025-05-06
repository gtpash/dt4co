import dolfin as dl
import ufl
import numpy as np

import hippylib as hp

from gemini.models import RDTumorVarf, RDTXTumorVarf, radioModel, chemoModel, mollifiedRDTumorVarf, mollifiedRDTXTumorVarf, PWRDTXTumorVarf, mollifiedPWRDTXTumorVarf
from gemini.dataModel import PatientData
from gemini.utils.mesh_utils import load_mesh, load_mesh_subs, report_mesh_info, check_mesh_dimension
from gemini.utils.model_utils import solveIndicators
from gemini.utils.parallel import root_print

class Experiment(object):
    """Set up an experiment's PDE problem.
    """
    
    def setupMesh(self, comm, mesh_fpath, zoff):
        """Set up the mesh for the problem.
        """
        raise NotImplementedError("Child class should implement method setupMesh")
    
    def setupFunctionSpaces(self, mesh):
        """Set up the function spaces for the problem.
        """
        raise NotImplementedError("Child class should implement method setupFunctionSpaces")
    
    
    def setupVariationalProblem(self, Vh, u0, t0, tf, sparam, moll, chi_gm):
        """Set up the variational problem defining the PDE model.
        """
        raise NotImplementedError("Child class should implement method setupVariationalProblem")
    
    
    def setupPrior(self, Vh):
        """Set up the prior for the experiment.
        """
        raise NotImplementedError("Child class should implement method setupPrior")
    
    
    def getBIPParameters(self):
        """Set up the parameter structure for the Newton-Krylov solver.
        """
        
        # Inverse problem parameters.
        bip_parameters = hp.ReducedSpaceNewtonCG_ParameterList()
        # bip_parameters["rel_tolerance"] = 1e-9
        # bip_parameters["abs_tolerance"] = 1e-12
        bip_parameters["max_iter"] = 50
        bip_parameters["globalization"] = "LS"
        bip_parameters["LS"]["max_backtracking_iter"] = 15
        bip_parameters["GN_iter"] = 5
        
        return bip_parameters


class rdExperiment(Experiment):
    """Class to hold the parameters and setup for the reaction-diffusion experiment.
    """
    def __init__(self):
        
        # Simulation parameters.
        self.STATE_DEGREE = 1        # degree of the state finite element space
        self.PARAM_DEGREE = 1        # degree of the parameter finite element space
        self.DELTA_T = 1.0           # time step [day]
        self.D0 = 0.068              # diffusion coefficient [mm^d / day]
        self.K0 = 0.059              # proliferation rate coefficient [1/day]
        self.RHO_K = 180.0           # proliferation rate correlation length [mm]
        self.RHO_D = 180.0           # diffusion correlation length [mm]
        self.VAR_K = 0.040           # variance of the proliferation rate
        self.VAR_D = 0.115           # variance of the diffusion coefficient
        self.NOISE = 0.0625          # noise level for the data (from [1])
        
        # Define the source term.
        self.f = dl.Constant(0.)
        
        # Define the boundary conditions.
        self.bc = []  # homogeneous Neumann for state
        self.bc0 = []  # homogeneous Neumann for adjoint
    
    
    def setupMesh(self, comm, mesh_fpath, zoff):
        mesh = load_mesh(comm, mesh_fpath)
    
        root_print(comm, f"Successfully loaded the mesh.")
        root_print(comm, f"There are {comm.size} process(es).")
        
        # Get the physical dimension of the mesh.
        zoff = check_mesh_dimension(mesh, zoff)
        
        report_mesh_info(mesh)
        return mesh, zoff
    
    
    def setupFunctionSpaces(self, mesh, mle: bool=False):
        #  Set up variational spaces for state and parameter.
        Vhu = dl.FunctionSpace(mesh, "Lagrange", self.STATE_DEGREE)
        
        if mle:
            Vhmi = dl.FunctionSpace(mesh, "Real", 0)  # use a constant function space for the parameter
        else:
            Vhmi = dl.FunctionSpace(mesh, "Lagrange", self.PARAM_DEGREE)
            
        mixed_element = ufl.MixedElement([Vhmi.ufl_element(), Vhmi.ufl_element()])
        Vhm = dl.FunctionSpace(mesh, mixed_element)  # todo: use vector function space?
        
        Vh = [Vhu, Vhm, Vhu]
        
        return Vh
    
    
    def setupVariationalProblem(self, Vh, u0, t0, tf, sparam, moll: bool=False):
                
        # Set the variational form for the forward model.
        if moll:
            varf = mollifiedRDTumorVarf(self.DELTA_T, lumped=False)
        else:
            varf = RDTumorVarf(self.DELTA_T, lumped=False)
        
        # Expecting solver parameters to be set from either CLI or .petscrc
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


    def setupPrior(self, Vh, mle: bool=False):
        # get the physical dimension of the mesh
        phys_dim = Vh[hp.STATE].mesh().topology().dim()
        
        # Set up the prior.
        mu = dl.Function(Vh[hp.PARAMETER])
        
        # assign the prior mean.
        mu.assign(dl.Constant([np.log(self.D0), np.log(self.K0)]))
        if mle:
            # maximum likelihood estimation, use a dummy prior.
            dummy_cov = np.eye(2)  # no preconditioning for the Newton solve
            mprior = hp.GaussianRealPrior(Vh[hp.PARAMETER], covariance=dummy_cov, mean=mu.vector())
            mprior.R.zero()  # zero out the regularization
        else:       
            diff_coeffs = hp.BiLaplacianComputeCoefficients(self.VAR_D, self.RHO_D, ndim=phys_dim)
            prolif_coeffs = hp.BiLaplacianComputeCoefficients(self.VAR_K, self.RHO_K, ndim=phys_dim)
            mprior = hp.VectorBiLaplacianPrior(Vh[hp.PARAMETER], [diff_coeffs[0], prolif_coeffs[0]], [diff_coeffs[1], prolif_coeffs[1]], mean=mu.vector(), robin_bc=True)

        return mprior


class rdtxExperiment(Experiment):
    """Class to hold the parameters and setup for the reaction-diffusion with (radio)therapy experiment.
    """
    def __init__(self, pinfo: PatientData):
        
        self.pinfo = pinfo  # patient information in the form of a PatientData object
        
        # Simulation parameters.
        self.STATE_DEGREE = 1        # degree of the state finite element space
        self.PARAM_DEGREE = 1        # degree of the parameter finite element space
        self.DELTA_T = 1.0           # time step [day]
        self.D0 = 0.068              # diffusion coefficient [mm^d / day]
        self.K0 = 0.059              # proliferation rate coefficient [1/day] (excluding radiotherapy effect)
        self.RHO_K = 180.0           # proliferation rate correlation length [mm]
        self.RHO_D = 180.0           # diffusion correlation length [mm]
        self.VAR_K = 0.040           # variance of the proliferation rate
        self.VAR_D = 0.115           # variance of the diffusion coefficient
        self.NOISE = 0.0625          # noise level for the data (from [1])
        
        # Define the radiotherapy term.
        self.ALPHA = 0.025              # LQ model alpha radiosensitivity parameter [1/day]
        self.alpha_beta_ratio = 10.0
        self.m_width = None             # do not use the mollified version
        
        # Define the chemotherapy term.
        self.BETA = 24. / 1.8           # clearance rate of the chemotherapy [1/day]
        
        # Define the boundary conditions.
        self.bc = []  # homogeneous Neumann for state
        self.bc0 = []  # homogeneous Neumann for adjoint
    
    
    def setupMesh(self, comm, mesh_fpath, zoff):
        mesh = load_mesh(comm, mesh_fpath)
    
        root_print(comm, f"Successfully loaded the mesh.")
        root_print(comm, f"There are {comm.size} process(es).")
        
        # Get the physical dimension of the mesh.
        zoff = check_mesh_dimension(mesh, zoff)
        
        report_mesh_info(mesh)
        return mesh, zoff
    
    
    def setupFunctionSpaces(self, mesh, mle: bool=False):
        #  Set up variational spaces for state and parameter.
        Vhu = dl.FunctionSpace(mesh, "Lagrange", self.STATE_DEGREE)
        
        if mle:
            Vhmi = dl.FunctionSpace(mesh, "Real", 0)  # use a constant function space for the parameter
        else:
            Vhmi = dl.FunctionSpace(mesh, "Lagrange", self.PARAM_DEGREE)
            
        # P1 x P1 for m_d, m_k
        mixed_element = ufl.MixedElement([Vhmi.ufl_element(), Vhmi.ufl_element()])
        Vhm = dl.FunctionSpace(mesh, mixed_element)
        
        Vh = [Vhu, Vhm, Vhu]
        
        return Vh
    
    
    def setupVariationalProblem(self, Vh, u0, t0, tf, sparam, moll: bool=False):
                
        # Set the variational form for the forward model.
        radio_model = radioModel(tx_days=self.pinfo.radio_days, tx_doses=self.pinfo.radio_doses, alpha=self.ALPHA, alpha_beta_ratio=self.alpha_beta_ratio, m_width=self.m_width)
        
        ct_sf = self.pinfo.chemo_effects[0]
        chemo_model = chemoModel(tx_days=self.pinfo.chemo_days, ct_effect=ct_sf, beta=self.BETA)
        
        # whether to use the mollified version of the model
        if moll:
            varf = mollifiedRDTXTumorVarf(self.DELTA_T, rtmodel=radio_model, ctmodel=chemo_model)
        else:
            varf = RDTXTumorVarf(self.DELTA_T, rtmodel=radio_model, ctmodel=chemo_model)
        
        # Expecting solver parameters to be set from either CLI or .petscrc
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


    def setupPrior(self, Vh, mle: bool=False):
        # get the physical dimension of the mesh
        phys_dim = Vh[hp.STATE].mesh().topology().dim()
        
        # Set up the prior.
        mu = dl.Function(Vh[hp.PARAMETER])
        
        # assign the prior mean.
        mu.assign(dl.Constant([np.log(self.D0), np.log(self.K0)]))
        if mle:
            # maximum likelihood estimation, use a dummy prior.
            dummy_cov = np.eye(2)  # no preconditioning for the Newton solve
            mprior = hp.GaussianRealPrior(Vh[hp.PARAMETER], covariance=dummy_cov, mean=mu.vector())
            mprior.R.zero()  # zero out the regularization
        else:       
            diff_coeffs = hp.BiLaplacianComputeCoefficients(self.VAR_D, self.RHO_D, ndim=phys_dim)
            prolif_coeffs = hp.BiLaplacianComputeCoefficients(self.VAR_K, self.RHO_K, ndim=phys_dim)
            mprior = hp.VectorBiLaplacianPrior(Vh[hp.PARAMETER], [diff_coeffs[0], prolif_coeffs[0]], [diff_coeffs[1], prolif_coeffs[1]], mean=mu.vector(), robin_bc=True)

        return mprior


class pwrdtxExperiment(Experiment):
    """Class to hold the parameters and setup for the piecewise reaction-diffusion with (radio)therapy experiment.
    """
    def __init__(self, pinfo: PatientData):
        
        self.pinfo = pinfo  # patient information in the form of a PatientData object
        
        # Simulation parameters.
        self.STATE_DEGREE = 1        # degree of the state finite element space
        self.PARAM_DEGREE = 1        # degree of the parameter finite element space
        self.DELTA_T = 1.0           # time step [day]
        self.DW = 0.068 * 1.5        # diffusion coefficient [mm^d / day]
        self.DG = self.DW / 5.       # diffusion coefficient (for gray matter) [mm^d / day]
        self.K0 = 0.059              # proliferation rate coefficient [1/day] (excluding radiotherapy effect)
        self.RHO_K = 180.0           # proliferation rate correlation length [mm]
        self.RHO_DG = 360.0          # diffusion correlation length in gray matter [mm]
        self.RHO_DW = 180.0          # diffusion correlation length in white matter [mm]
        self.VAR_K = 0.040           # variance of the proliferation rate
        self.VAR_DW = 0.115          # variance of the diffusion coefficient (for white matter)
        self.VAR_DG = 0.115          # variance of the diffusion coefficient (for gray matter)
        self.NOISE = 0.0625          # noise level for the data (from [1])
        
        # Define the radiotherapy term.
        self.ALPHA = 0.025              # LQ model alpha radiosensitivity parameter [1/day]
        self.alpha_beta_ratio = 10.0
        self.m_width = None             # do not use the mollified version
        
        # Define the chemotherapy term.
        self.BETA = 24. / 1.8           # clearance rate of the chemotherapy [1/day]
        
        # Define the boundary conditions.
        self.bc = []   # homogeneous Neumann for state
        self.bc0 = []  # homogeneous Neumann for adjoint
        
        # tissue segmentation (to be set up during setupMesh)
        self.chi_gm = None
    
    
    def setupMesh(self, comm, mesh_fpath, zoff):
        mesh, subs, _ = load_mesh_subs(comm, mesh_fpath)
    
        root_print(comm, f"Successfully loaded the mesh.")
        root_print(comm, f"There are {comm.size} process(es).")
        
        # Get the physical dimension of the mesh.
        zoff = check_mesh_dimension(mesh, zoff)
        
        report_mesh_info(mesh)
        
        # solve for the tissue segmentation indicator function
        chi_gm = solveIndicators(mesh, subs, 1)
        self.chi_gm = chi_gm
        
        return mesh, zoff
    
    
    def setupFunctionSpaces(self, mesh, mle: bool=False):
        #  Set up variational spaces for state and parameter.
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
    
    
    def setupVariationalProblem(self, Vh, u0, t0, tf, sparam, moll: bool=False):
                
        # Set the variational form for the forward model.
        radio_model = radioModel(tx_days=self.pinfo.radio_days, tx_doses=self.pinfo.radio_doses, alpha=self.ALPHA, alpha_beta_ratio=self.alpha_beta_ratio, m_width=self.m_width)
        
        ct_sf = self.pinfo.chemo_effects[0]
        chemo_model = chemoModel(tx_days=self.pinfo.chemo_days, ct_effect=ct_sf, beta=self.BETA)
        
        # whether to use the mollified version of the model
        if moll:
            varf = mollifiedPWRDTXTumorVarf(self.DELTA_T, rtmodel=radio_model, ctmodel=chemo_model, chi_gm=self.chi_gm)
        else:
            varf = PWRDTXTumorVarf(self.DELTA_T, rtmodel=radio_model, ctmodel=chemo_model, chi_gm=self.chi_gm)
        
        # Expecting solver parameters to be set from either CLI or .petscrc
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


    def setupPrior(self, Vh, mle: bool=False):
        # get the physical dimension of the mesh
        phys_dim = Vh[hp.STATE].mesh().topology().dim()
        
        # Set up the prior.
        mu = dl.Function(Vh[hp.PARAMETER])
        
        # assign the prior mean (m_dg, m_dw, m_k).
        mu.assign(dl.Constant([np.log(self.DG), np.log(self.DW), np.log(self.K0)]))
        if mle:
            # maximum likelihood estimation, use a dummy prior.
            dummy_cov = np.eye(3)  # no preconditioning for the Newton solve
            mprior = hp.GaussianRealPrior(Vh[hp.PARAMETER], covariance=dummy_cov, mean=mu.vector())
            mprior.R.zero()  # zero out the regularization
        else:       
            dg_coeffs = hp.BiLaplacianComputeCoefficients(self.VAR_DG, self.RHO_DG, ndim=phys_dim)
            dw_coeffs = hp.BiLaplacianComputeCoefficients(self.VAR_DW, self.RHO_DW, ndim=phys_dim)
            prolif_coeffs = hp.BiLaplacianComputeCoefficients(self.VAR_K, self.RHO_K, ndim=phys_dim)
            mprior = hp.VectorBiLaplacianPrior(Vh[hp.PARAMETER], [dg_coeffs[0], dw_coeffs[0], prolif_coeffs[0]], [dg_coeffs[1], dw_coeffs[1], prolif_coeffs[1]], mean=mu.vector(), robin_bc=True)

        return mprior
        

class ExperimentFactory:
    """Class to ease experiment creation.
    """
    def __init__(self, pinfo: PatientData):
        self.pinfo = pinfo
        self.experiment_map = {
            'rd': rdExperiment,
            'rdtx': rdtxExperiment,
            'pwrdtx': pwrdtxExperiment,
            # todo: add others
        }
        
    def get_experiment(self, experiment_name) -> Experiment:
        if experiment_name in self.experiment_map:
            return self.experiment_map[experiment_name]() if experiment_name == 'rd' else self.experiment_map[experiment_name](self.pinfo)
        else:
            raise ValueError(f"Experiment '{experiment_name}' not recognized.")

# --------------------------------------------------------------------------------
# [1] Liang, Baoshan, et al. "Bayesian inference of tissue heterogeneity for individualized prediction of glioma growth." IEEE Transactions on Medical Imaging 42.10 (2023): 2865-2875.
