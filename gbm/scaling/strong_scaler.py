################################################################################
# 
# This is the driver script for the strong scaling study.
# 
################################################################################

import os
import sys
import argparse
import time
import dolfin as dl
from mpi4py import MPI  # MUST be imported AFTER dolfin
import numpy as np

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
import hippylib as hp

sys.path.append(os.path.join(os.getenv("DT4CO_PATH"), "src"))
from dt4co.utils.mesh_utils import report_mesh_info
from dt4co.utils.model_utils import samplePrior
from dt4co.utils.parallel import root_print

from problem_setup import setupProblem, setupAdjoint

def main(args):
    ############################################################
    # 0. General setup.
    ############################################################
    # Average brain dimensions from [1] are approximately:
    # 140 mm wide, 167 mm long, and 93 mm high.
    # [1] https://faculty.washington.edu/chudler/facts.html
    
    vv = 100.0  # rough size [mm]
    
    # MPI setup.
    # COMM = MPI.COMM_WORLD
    COMM = dl.MPI.comm_world
    rank = COMM.rank
    nproc = COMM.size
    
    root_print(COMM, f"There are {nproc} process(es).")
    
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)
    LOGFILE = os.path.join(OUTDIR, args.log)

    ############################################################
    # 1. Generate the mesh.
    ############################################################
    # Build isotropic mesh to be distributed.
    work = args.work
    ncells = np.rint(np.cbrt(work)).astype(int)  # isotropic
    box = dl.BoxMesh(COMM, dl.Point(0, 0, 0), dl.Point(vv, vv, vv), ncells, ncells, ncells)
    report_mesh_info(box)
    
    ndof = COMM.allreduce(box.num_vertices(), op=MPI.SUM)
    
    ############################################################
    # 2. Set up the problem.
    ############################################################
        
    pde, mprior = setupProblem(box)
    
    ############################################################
    # 3. Solve the forward problem.
    ############################################################
    m0 = samplePrior(mprior)    # sample from the prior
    u = pde.generate_state()    # vector to store state
    a = pde.generate_adjoint()  # vector to store adjoint
    x0 = [u, m0, a]
    
    root_print(COMM, "Beginning forward solve.")
    
    start = time.perf_counter()
    pde.solveFwd(x0[hp.STATE], x0)
    end = time.perf_counter() - start
    
    # Append result to file.
    if rank == 0:
        with open(LOGFILE, 'a') as fwd_log:
            fwd_log.write(f"{nproc}\t{ndof}\t{end}\n")
    
    ############################################################
    # 4. Optionally, solve the adjoint problem.
    ############################################################
    if args.adjoint:
        LOGFILE = os.path.join(OUTDIR, f"adjoint_{args.log}")  # update the log file
        
        misfit = setupAdjoint(pde, x0)
        model = hp.Model(pde, mprior, misfit)
        
        root_print(COMM, "Beginning forward solve.")
        
        start = time.perf_counter()
        model.solveAdj(x0[hp.ADJOINT], x0)
        end = time.perf_counter() - start
        
        # Append result to file.
        if rank == 0:
            with open(LOGFILE, 'a') as adj_log:
                adj_log.write(f"{nproc}\t{ndof}\t{end}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate a series of meshes for scaling tests.")
    
    # output options
    parser.add_argument("--log", type=str, help="Log file for output.")
    parser.add_argument("--outdir", type=str, default="logs", help="Where to save output.")
    parser.add_argument("--adjoint", action=argparse.BooleanOptionalAction, default=False, help="Test the scaling of the adjoint solve?")
    
    # work options
    parser.add_argument("--work", type=int, default=1e7, help="Number of DOF.")
    
    args = parser.parse_args()
    main(args)
