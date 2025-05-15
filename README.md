# `dt4co`
Developing digital twins for computational oncology.

## Code Description
This repository contains codes to support the development of predictive digital twins for oncology. In particular, these codes are to the modeling and simulation of glioblastoma multiforme / high grade gliomas. This repository spans data preprocessing, implementation of the forward model, quantification of uncertainty, and forward uncertainty propagation to quanities of interest. The pipeline begins with the preparation of longitudinal medical imaging and meshing of the biological domain. A low-rank approximation to the posterior is computed and uncertainty is propagated through the forward model. There are two major applications provided in this repository:
1. An implementation for a cohort of patients and the requisite components of the pipeline are implemented in the `gbm` subdirectory (see the [`README`](./gbm/README.md) for more information).
2. Simulation studies are useful for isolating individual parameters of the experimental setup and an implementation is provided in the `synth` directory (see the [`README`](./synth/README.md) for more information). In this case, the tradeoff between imaging frequency and prediction accuracy was studied.

Visualization is primarily handled with [ParaView](https://www.paraview.org/) and `.pvsm` templates to support analysis are provided in the `paraview` subdirectory. Source codes that provide functionality to drive the aforementioned applications are in the `src` subdirectory.

## Major Components
The major components of this work are: 

1. Medical imaging data preparation
2. Volume mesh generation
3. Scalable implementation of the forward model
4. Efficient approximation of the posterior
5. Forward uncertainty propagation
6. Quantity of interest computation

## Software Installation
Docker was used to ensure portability of the codes and there are three major docker images used in this project:
- Data Preparation [[Dockerfile]](./gbm/preprocessing/docker/Dockerfile)
- Mesh Generation [[Dockerfile]](./gbm/meshing/docker/Dockerfile)
- Computations [[Dockerfile]](./gbm/docker/Dockerfile)

This does not prohibit a user from installing all the software requirements in one environment, but rather provides example siloed implementations for HPC systems. To utilize the source codes, ensure that the appropriate requirements are installed, set the `DT4CO_PATH` environmental variable to the directory where this code has been cloned, and add the `src` directory to your path:
```
sys.path.append( os.path.join( os.environ.get('DT4CO_PATH'), "src" ) )
```

## Reference
If you find this library useful in your research, please consider citing the following:
```
@misc{pash2025predictivedigitaltwinsquantified,
      title={Predictive Digital Twins with Quantified Uncertainty for Patient-Specific Decision Making in Oncology}, 
      author={Graham Pash and Umberto Villa and David A. Hormuth II and Thomas E. Yankeelov and Karen Willcox},
      year={2025},
      eprint={2505.08927},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2505.08927}, 
}
```