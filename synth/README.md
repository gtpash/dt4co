# Synthetic Patient Simulation Studies
This directory contains scripts required to run simulation studies on a "synthetic" patient. The geometry is taken to be that of a real patient that has been appropriately pre-processed with the pipeline laid out in this code base. In particular, patient `101` from the UPENN-GBM dataset. The steps of the study are as follows,

1. Generate observational data from a known underlying model and corrupt measurements with Gaussian noise.
- Data generation is handled by the script `generate_synthetic_observations.py`.
2. Solve the deterministic inverse problem for the MAP point reconstruction and generate the Laplace approximation to the posterior.
- The inverse problem is run by the `run_synth_bip.py` script.
3. Draw samples from the prior and posterior, either by sampling the Laplace approximation or via Markov Chain Monte Carlo (MCMC).
- To draw samples from the prior and the Laplace approximation, please use `draw_la_samples.py`.
- To draw samples from the posterior with MCMC, please use `draw_mcmc_samples.py`.
4. Compute the pushforward of the samples.
- To compute the pushfowards, please use `run_fwd_prop.py`.
- To concatenate the results from the forward propagation run, use `concat_fwd_prop.py`.
5. Compute the quantities of interest for comparison.
- Various quantities of interest are computed with the `compute_qoi.py` script.

The study was executed on TACC's `frontera` supercomputer and the associated jobfiles may be found in the `jobs` subdirectory. Assuming that the geometry was processed with `FreeSurfer` as previously described, the `pp_VBG.slurm` jobfile shows how to run the necessary registration pipeline. The implementation of the above pipeline would require execution of the following scripts:
```
generate_synth_data.slurm (step 1)
run_synth_rd_bip.slurm (step 2)
run_synth_prop.slurm (steps 3+4)
compute_qoi.slurm (step 5)
```

Note(s):
- We utilize an observation operator to convert from state measurements (on the mesh) to the discrete data-space (voxel images)
- We use a coarser mesh to evolve the dynamics in the inverse problem so as to avoid committing an "inverse crime".
- To test the BIP for a tumor in a box example, consider `run_box_bip.py`.
- To utilize a continuous observation operator, use `run_cont_synth_bip.py`.
- To compute the mass-matrix orthogonal eigenvectors (instead of the prior-preconditioned), use `mass_orth_misfit.py`.
