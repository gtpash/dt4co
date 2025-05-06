# Glioblastoma Multiforme (GBM)
We demonstrate application of the digital twin paradigm to the modeling of high grade gliomas.

## Running the pipeline
The steps of the pipleine are:

1. Data preparation
2. Volume mesh generation
3. Computation of the low-rank approximation to the posterior
4. Forward uncertainty propagation
5. Quantity of interest computation

Job files to run the pipeline on the TACC `frontera` supercomputer are available in the `jobs` subdirectory. The `cohort_submitter.sh` script may be used to rapidly submit jobs for multiple patients in the cohort (see the script for valid jobs over which submission may be parallelized). The following job files are used to run the pipeline
- `run_VBG.slurm` and `pp_VBG.slurm`: data preparation
- `run_cohort_meshing.slurm`: mesh generation
- `run_bip.slurm`: computation of the Laplace approximation
- `run_fwd_prop.slurm`: propagation of uncertainty
- `run_qoi.slurm` (`run_cohort_qoi.slurm` for the cohort): quantity of interest computation

### Software Installation
A `Dockerfile` is provided in the `docker` subdirectory. This container was used on TACC `frontera` system.

## Descriptions
### Data preparation
It is important to first register longitudinal medical imaging data to a common reference coordinate system. Codes to support this effort are provided in `preprocessing` subdirectory and more information is available in the [`README`](./preprocessing/README.md).

### Volume mesh generation
The finite element method will be used for the discretization and solution of the governing equations. We cover mesh generation for the complex biological domain in the `meshing` subdirectory. More information is available in the [`README`](./meshing/README.md).

### Computation of the low-rank approximation to the posterior
We leverage [`hIPPYlib`](https://github.com/hippylib/hippylib) for scalable computation of the Laplace approximation to the posterior. In turn, this enables rapid sampling for forward propagation. We compute this approximation with the `run_bip.py` script. 

### Propagation of uncertainty
The prior / posterior uncertainty is propagated through the forward model by sampling with `draw_la_samples.py` and then repeatedly solving the forward model. Samples are concatenated and stored with `concat_fwd_prop.py`.

### Quantities of interest
The `compute_qoi.py` script is used to compute quantities of interest for the

### Scaling of the forward model
Strong and weak scaling of the forward model is studied and reported in the `scaling` subdirectory.

## Miscellaneous
- To ensure that the adjoint gradient and Hessian actions of the governing equations are implemented correctly, one may use the `run_verify_model.py` script.
- The `run_rd_forward.py` script may be used to run the forward model once for a fixed parameter.
- `PETSc` command line options are supported. The forward model requires the `fwd` prefix, e.g. Example: `-fwd_ksp_type gmres`. An example `.petscrc` is provided in this directory.
- `run_mle.slurm` computes a maximum likelihood estimate of the (scalar) model parameters, for development of a cohort prior.
- `run_qoi_pipeline.py` an alternative quantity of interest pipeline utilizing `run_rd_forward.py`.
- `pp_VBG.slurm` will utilize the `viz_registration.py` script to project registered longitudinal data onto a full brain mesh for visualization.
