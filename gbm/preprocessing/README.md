# Data Preparation
The pre-processing stage of this work consists of the necessary steps to enable mesh generation for a patient-specific computational domain: anatomic segmentation, intra-visit registration of different imaging modalities, inter-visit registration of longtiduinal patient data. An overview of the pipeline is presented below.

![An overview of the pipeline.](pipeline.png)

## Software
The following software are required:
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) for tissue segmentation and surface generation. Installation instructions are [readily available](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) and the _Mardal et. al._ book is an excellent introductory reference. FreeSurfer also comes with FreeView, which is a useful tool for visualizing NIfTI images and FreeSurfer generated surfaces.
- [SimpleITK](https://simpleitk.org/) for application of OncoHabitats ANTsX registrations. There is a `conda` installation option, however I have found that [building from source](https://simpleitk.readthedocs.io/en/master/building.html#building-simpleitk) is easy and more robust.
- [Elastix](https://elastix.dev/) or [`itk-elastix`](https://github.com/InsightSoftwareConsortium/ITKElastix) for registration. Compiled binaries are [readily available](https://elastix.dev/download.php) and the manual provides excellent documentation. `itk-elastix` can easily be installed into a Python environment.
- [`dcm2niix`](https://github.com/rordenlab/dcm2niix) for conversion of DICOM data to NIfTI format.
- [Paraview](https://www.paraview.org/) for visualization of NIfTI data as well as the generated meshes.

A `Dockerfile` is provided in the `docker` repository for use on TACC machines, and may be modified for another. Note that you will need to obtain a `license.txt` file from `FreeSurfer` and place it in this directory so that `FreeSurfer` will build correctly.

### Miscellaneous Resources
Neuroimaging has a rich and complex ecosystem of software supporting various research objectives. A non-exhaustive list of resources for understanding the "lay of the land" as well as general image registration concepts are:
- [Andy's Brain Book](https://andysbrainbook.readthedocs.io/en/latest/)
- [FreeSurfer's Coordinate Systems](https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems)
- [Blogpost on coordinate systems](http://www.grahamwideman.com/gw/brain/fs/coords/fscoords.htm)
- [Horus](https://horosproject.org/) is a good software for viewing DICOM data.
- [ANTs](https://github.com/ANTsX/ANTs) is an alternative toolchain for computing registrations, tissue segmentations, etc.
- [Structural MRI Carpentries Tutorial](https://carpentries-incubator.github.io/SDC-BIDS-sMRI/aio/index.html)
- [Nilearn](https://nilearn.github.io/stable/index.html#)
- *Mathematical Modeling of the Human Brain: From Magnetic Resonance Imaging to Finite Element Simulation* by Kent-André Mardal, Marie E. Rognes, Travis B. Thompson, and Lars Magnus Valnes.

## Dataset Descriptions
We primarily consider two datasets publically available through the [the Cancer Imaging Archive](https://www.cancerimagingarchive.net/).
- [University of Pennsylvania glioblastoma cohort](https://www.nature.com/articles/s41597-022-01560-7) (UPENN-GBM)
- [Ivy Glioblastoma Atlas Project](https://www.google.com/search?q=ivy+gap&oq=ivy+gap&gs_lcrp=EgZjaHJvbWUqBggAEEUYOzIGCAAQRRg7MggIARAAGBYYHjIICAIQABgWGB4yCAgDEAAYFhgeMggIBBAAGBYYHjIGCAUQRRg9MgYIBhBFGD0yBggHEEUYPdIBBzk5OWowajGoAgCwAgA&sourceid=chrome&ie=UTF-8) (IvyGAP)

### UPENN-GBM Dataset
Four sets of anatomical images are provided for each patient:

- FLAIR
- T1-GD
- T1
- T2

These are provided as zipped NIfTI files. These files are available in either stripped or unstripped form (where the skull has been removed from the imaging). For the purposes of using FreeSurfer, I have found that using the unstripped images is preferable as one of the steps in the segmentation process is to strip the images. Additionally, radiologist derived tumor segmentations are provided for each patient as a zipped NIfTI file.

Note that data is only available at baseline (diagnosis) for each patient. No longitudinal data is provided.

### IvyGAP Dataset
The IvyGAP dataset contains longitudinal studies for the cohort of patients. At each imaging date, various sequences of imaging are performed and provided in [DICOM](https://www.dicomstandard.org/) format. While the exact sequence varies session to session, once can expect a T1, T2, T1-contrast, and T2FLAIR scan to be available.

Tumor segmentations are **not** provided. To generate the segmentations, the [OncoHabitats](https://www.oncohabitats.upv.es/) service is used to process the DICOM data and provide segmentations. Simply submit the T1, T2, T1-C, and FLAIR scans for processing, which takes around 30 minutes per job.

When dealing with longitudinal data, one must deal with the registration problem, i.e. correlating structures in a scan at a later date to structures in a scan at a previous date. We will assume that the brain geometry does not substantially change over the course of the study and thus will generate the computational geometry from the initial scan, and map the tumor segmentations onto this geometry at future time instances. If the resection cavity is large, this may not be appropriate and the data may be co-registered to a later scan.

# Description of Pre-Processing Steps

## Working with a common data format: NIfTI
Users are encouraged to utilize the [NIfTI](https://nifti.nimh.nih.gov) format. The first step in the pre-processing pipeline described here is to convert all DICOM data to NIfTI format. This could be achieved manually using a tool such as `dcm2niix`, however a conveince script `convertDCM2NII.sh` is provided.

## Anatomical Segmentation
The first step to generating a computational mesh is to segment out the various tissues comprising the brain. To perform all of the segmentations with FreeSurfer, simply run:

```
recon-all -subject NAME -i /path/to/data -all
```

This will perform **all** segmentations on the input file specified and save the segmentations to `$SUBJECTS_DIR/NAME`. Note that these segementations may be refined with T2 imaging, by also specifying the path to the T2 image. Refer to [the FreeSurfer Wiki](https://surfer.nmr.mgh.harvard.edu/fswiki) for more information on the [recon-all command](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all) and its [outputs](https://surfer.nmr.mgh.harvard.edu/fswiki/ReconAllOutputFiles). This step will take roughly 6 hours to complete, but only needs to be performed once per patient.

**NOTE:** It is advised that one use the NIfTI version of the T1 image for consistency.

To work with the segmentations, we must convert the binary surface file into the STL format. This is accomplished with the command `mris_convert`. Segmentation fo the ventricles is possible by post-processing the `recon-all` outputs. 

Postprocessing of the FreeSurfer outputs is handled with the `postprocFreeSurfer.sh` bash script. This script performs the following steps:
1. Segmentation of the ventricles
2. Shift pial/white/ventricular surfaces to align with the FreeSurfer origin by applying the `c_RAS`
3. Convert FreeSurfer surfaces to STL files

An alternative workflow is somewhat developed in `postprocFreeSurferNative.sh`. This script performs the following steps:
1. Registration of the functional image with the conformed volume (native coordinates -> FreeSurfer coordinates)
2. Apply the transformation to the FreeSurfer surfaces
3. Segmentation of the ventricles

However, these coordinates were found to be more difficult to work with. There is some limited doucmentation on this matter, please reach out for more information.

## (IvyGAP) Longitudinal Registration
An example pre-processing sequence for the [IvyGAP](https://glioblastoma.alleninstitute.org/) dataset (available via [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/ivygap/)) is implemented both with binary versions of `elastix` and without in the `IvyGAP/runElastixPipeline.sh` bash script and `IvyGAP/runLongitudinalRegistration.py` python script, respectively. These scripts prepares the patient data through the following steps:
1. Registration of the reference T1 image to the FreeSurfer T1 volume.
2. Rigid registration of the baseline T1 image to each longitudinal T1 image
3. Deformable registration of the baseline T1 image to longitudinal T1 image
4. Registration of the OncoHabitats segmentation to the intra-visit T1
5. Application of registration transformation to longitudinal segmentations to bring them to baseline T1 space

**Assumptions:**
Some manual data preparation steps are required.
- The anatomical imaging DICOM directories are renamed for consistency to: T1, T2, T2FLAIR, T1C, DWI, ADC, DTI
- OncoHabitats is used to generate tumor segmentations and that the results of that analysis are stored in an `oncohabitats` directory.
- The patient should have directories for each imaging date formatted as YYYY_MM_DD

# Summary
To process a new patient, the following commands should be run to complete the pre-processing:
```
recon-all -subject NAME -i /path/to/T1 -all
./postprocFreeSurfer.sh NAME /path/to/T1
python3 IvyGAP/runLongitudinalRegistration.py --pdir /path/to/patient/data --subjid FreeSurferSubjectID --elxparams /path/to/elastix/parameters
python3 computeCellularity.py --pdir /path/to/patient/data
```

Visualization of the tumor segmentations on the mesh can be accomplished by calling
```
python3 ../viz_tumor_segmentations.py
```

**NOTE:** Be sure to update the patient-specific directories in the `runElastixPipeline.sh` script.

# For Advanced Users

It is known that FreeSurfer struggles to generate accurate anatomical segmentations in the presence of large lesions. Our preferred solution to handle this case is to use the [`KUL_VBG`](https://github.com/KUL-Radneuron/KUL_VBG) tool to first "graft" healthy tissue in place of the lesion and then run FreeSurfer's normal `recon-all` call on the "filled" image.

> Radwan, Ahmed M., Louise Emsell, Jeroen Blommaert, Andrey Zhylka, Silvia Kovacs, Tom Theys, Nico Sollmann, Patrick Dupont, and Stefan Sunaert. "Virtual brain grafting: Enabling whole brain parcellation in the presence of large lesions." NeuroImage 229 (2021): 117731.

An example workflow would be:
```
# generate the "filled" image
KUL_VBG.sh -S subID -a /path/to/subjects/T1/image -z T1 -o /path/to/store/KUL_VBG/results/ -l /path/to/ROI/for/masking/ -P 1 -v

# perform anatomic segmentation on the "filled" image
recon-all -subject subID -i /path/to/KUL_VBG/result/image/

# registration pipeline may continue as normal...
```

Please follow the installation instructions on the `KUL_VBG` repository for your machine. For convenience a Dockerfile is provided that will install the necessary dependencies. Be sure to update the FreeSurfer license file to your own.
