import os
import sys
import argparse

import dolfin as dl
import numpy as np

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
import hippylib as hp

sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
from dt4co.synth import synthExperiment
from dt4co.utils.mesh_utils import report_mesh_info, load_mesh
from dt4co.utils.fenics_io import read_mv_from_h5, write_mv_to_xdmf
from dt4co.utils.parallel import root_print


class OffDiagLowRankOperator:
    """
    This class implements the action of the off-diagonal blocks of a low rank operator.
    """

    def __init__(self, d, U1, U2, my_init_vector=None):
        """
        Construct the low rank operator given :code:`d` and :code:`U`.
        """
        self.d = d
        self.U1 = U1
        self.U2 = U2
        self.my_init_vector = my_init_vector
        self.help1 = dl.Vector(U1[0].mpi_comm())
        self.init_vector(self.help1, 0)
        self.help2 = dl.Vector(U2[0].mpi_comm())
        self.init_vector(self.help2, 0)

    def init_vector(self, x, dim):
        """
        Initialize :code:`x` to be compatible with the range (:code:`dim=0`) or domain (:code:`dim=1`) of :code:`A`.
        """
        assert self.my_init_vector is not None
        self.my_init_vector(x, dim)

    def mult(self, x, y):
        """
        Compute :math:`y = Ax = ( U1 D U2^T + U2 D U1^T ) x`
        """

        # compute the low rank action using the factorized form of the operator, i.e. first compute V^T x, then do the elementwise multiplication with d, then compute U (D (V^T x)).
        U1tx = self.U1.dot_v(x)
        U2tx = self.U2.dot_v(x)
        dU1tx = self.d * U1tx  # elementwise mult
        dU2tx = self.d * U2tx  # elementwise mult

        self.help1.zero()
        self.U2.reduce(self.help1, dU1tx)
        self.help2.zero()
        self.U1.reduce(self.help2, dU2tx)

        y.zero()
        y.axpy(1.0, self.help1)
        y.axpy(1.0, self.help2)


def main(args) -> None:

    # -----------------------------------------------------------
    # 0. Unpack input arguments.
    # -----------------------------------------------------------
    SEP = "\n" + "#" * 80 + "\n"

    COMM = dl.MPI.comm_world

    # unpack arguments
    EVAL_FILE = args.eval
    EVEC_FILE = args.evec
    MAP_FILE = args.map
    MESH_FPATH = args.mesh
    NMODES = args.nmodes

    os.makedirs("output", exist_ok=True)

    # -----------------------------------------------------------
    # 1. Set up necessary experiment objects.
    # -----------------------------------------------------------

    root_print(COMM, SEP)
    root_print(COMM, f"Loading in the mesh...")

    mesh = load_mesh(COMM, MESH_FPATH)

    root_print(COMM, f"Successfully loaded the mesh.")
    report_mesh_info(mesh)

    root_print(COMM, f"Setting up experiment and function spaces.")
    exp = synthExperiment()

    #  Set up variational spaces for state and parameter.
    Vh = exp.setupBIPFunctionSpaces(mesh, mle=False)
    mprior = exp.setupPrior(Vh)
    mfun = dl.Function(Vh[hp.PARAMETER])

    # -----------------------------------------------------------
    # 2. Read back the eigenpairs, set up the Laplace approximation.
    # -----------------------------------------------------------

    root_print(COMM, SEP)
    root_print(COMM, f"Reading in the eigenpairs...")
    root_print(COMM, f"Eigenvalue file:\t{EVAL_FILE}")
    root_print(COMM, f"Eigenvector file:\t{EVEC_FILE}")

    # eigenvalues.
    evals = np.loadtxt(EVAL_FILE)
    evals = evals[:NMODES] if NMODES > 0 else evals  # subset if necessary
    num_evec = len(evals)

    # set up mulitvector to store the eigenvectors.
    evec = hp.MultiVector(mfun.vector(), num_evec)
    read_mv_from_h5(COMM, evec, Vh[hp.PARAMETER], EVEC_FILE, name="gen_evec")

    # read back the MAP point.
    mmap = hp.MultiVector(mfun.vector(), 1)
    read_mv_from_h5(COMM, mmap, Vh[hp.PARAMETER], MAP_FILE, name=["map"])

    post = hp.GaussianLRPosterior(mprior, evals, evec)
    post.mean = mmap[0]

    # -----------------------------------------------------------
    # break out the eigenvectors into their D and K components
    # -----------------------------------------------------------
    assigner, Vhi = exp.setupFunctionAssigner(Vh[hp.PARAMETER])

    # split the eigenvectors into their components
    UD = hp.MultiVector(mfun.vector(), num_evec)
    UK = hp.MultiVector(mfun.vector(), num_evec)

    full_out = dl.Function(Vh[hp.PARAMETER])
    tmp_full = dl.Function(Vh[hp.PARAMETER])
    tmp_D = dl.Function(Vhi)
    tmp_K = dl.Function(Vhi)

    for i in range(num_evec):
        tmp_full.vector().zero()
        tmp_full.vector().axpy(1.0, evec[i])

        # first do the D component
        full_out.vector().zero()
        tmp_D.vector().zero()
        tmp_D.vector().axpy(1.0, tmp_full.sub(0, deepcopy=True).vector())
        tmp_K.vector().zero()
        assigner.assign(full_out, [tmp_D, tmp_K])

        UD[i].zero()
        UD[i].axpy(1.0, full_out.vector())

        # then do the K component
        full_out.vector().zero()
        tmp_D.vector().zero()
        tmp_K.vector().zero()
        tmp_K.vector().axpy(1.0, tmp_full.sub(1, deepcopy=True).vector())
        assigner.assign(full_out, [tmp_D, tmp_K])

        UK[i].zero()
        UK[i].axpy(1.0, full_out.vector())

    if args.verbose:
        root_print(COMM, f"Writing out the eigenvector components to file...")
        write_mv_to_xdmf(COMM, UD, Vh[hp.PARAMETER], os.path.join("output", "evec_D.xdmf"))
        write_mv_to_xdmf(COMM, UK, Vh[hp.PARAMETER], os.path.join("output", "evec_K.xdmf"))

    # combine the D and K components into a single multivector for use in the trace computations.
    U_DK = hp.MultiVector(mfun.vector(), num_evec + num_evec)
    for i in range(num_evec):
        U_DK[i].zero()
        U_DK[i].axpy(1.0, UD[i])
        U_DK[num_evec + i].zero()
        U_DK[num_evec + i].axpy(1.0, UK[i])

    if args.verbose:
        root_print(COMM, f"Writing out the combined eigenvectors to file...")
        write_mv_to_xdmf(COMM, U_DK, Vh[hp.PARAMETER], os.path.join("output", "evec_DK.xdmf"))

    # -----------------------------------------------------------
    # solve the generalized eigenvalue problem, compute the mutual information.
    # -----------------------------------------------------------

    d_DK = np.concatenate([evals, evals])  # duplicate the eigenvalues for the combined multivector
    Hlr_diag = hp.LowRankHessian(post.prior, d_DK, U_DK)
    post_diag = Hlr_diag.LowRankHinv

    # re-orthogonalize the eigenvectors with respect to the prior covariance
    kk = num_evec  # re-orthogonalize with respect to the number of modes used in the low rank approximation
    pp = 10
    Omega = hp.MultiVector(mfun.vector(), kk + pp)
    hp.parRandom.normal(1.0, Omega)
    d_post_diag, U_post_diag = hp.doublePassG(post_diag, mprior.R, mprior.Rsolver, Omega, kk, s=1, check=False)

    # Correction weights: -1.0 * lambda_i / (lambda_i + 1), with lamda_i from the prior-preconditioned Hessian
    corr_weights = -1.0 * evals / (evals + np.ones_like(evals))
    post_off_diag = OffDiagLowRankOperator(corr_weights, UD, UK, my_init_vector=post.init_vector)

    reorth_Hlr_diag = hp.LowRankHessian(post.prior, d_post_diag, U_post_diag)
    reorth_post_diag = reorth_Hlr_diag.LowRankHinv

    kk = 16
    pp = 10
    Omega = hp.MultiVector(mfun.vector(), kk + pp)
    hp.parRandom.normal(1.0, Omega)
    d_mi, U_mi = hp.doublePassG(post_off_diag, reorth_post_diag, reorth_post_diag, Omega, k=kk, s=1, check=False)

    if COMM.rank == 0:
        np.save(os.path.join("output", "d_mi.npy"), d_mi)

    MI = -0.5 * np.sum(np.log(1 + d_mi))
    root_print(COMM, f"Mutual Information: {MI}")

    r_linfoot = np.sqrt(1 - np.exp(-2 * MI))
    root_print(COMM, f"Linfoot's r-value:{r_linfoot:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate samples from the prior and Laplace approximation of the posterior.")

    # data inputs.
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--eval", type=str, required=True, help="File containing the eigenvalues.")
    parser.add_argument("--evec", type=str, required=True, help="File containing the eigenvectors.")
    parser.add_argument("--map", type=str, required=True, help="File containing the MAP point.")

    # modeling inputs.
    parser.add_argument("--nmodes", type=int, default=-1, help="Number of modes to use.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Whether or not to print additional information and write out the eigenvector components.")

    args = parser.parse_args()
    main(args)
