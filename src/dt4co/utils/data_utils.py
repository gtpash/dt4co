import os
import math
from datetime import datetime

import numpy as np
import scipy
import nibabel
from nibabel.affines import apply_affine
import SimpleITK as sitk
import dolfin as dl

from hippylib import NumpyScalarExpression3D, NumpyVectorExpression3D, NumpyScalarExpression2D, numpy2MeshFunction
from hippylib import ContinuousStateObservation, DiscreteStateObservation, MisfitTD
from hippylib import assemblePointwiseObservation
from hippylib import parRandom

from .parallel import gather_to_zero, numpy2Vec

# ----------------------------------------------
# Helper functions.
# ----------------------------------------------

def get_vox2ras(nii: nibabel.nifti1) -> np.ndarray:
    """Get the voxel to RAS transform from NIfTI image.
    Method from the forums: https://neurostars.org/t/get-voxel-to-ras-transformation-from-nifti-file/4549
    
    Args:
        t1: NIfTI image loaded by nibabel.

    Returns:
        np.ndarray: Voxel to RAS transformation.
    """
    ds = nii.header.get_zooms()[:3]
    ns = np.array(nii.shape[:3]) * ds / 2.0
    vox2ras = np.array([[ds[0], 0, 0, -ns[0]],
                       [0, ds[1], 0, -ns[1]],
                       [0, 0, ds[2], -ns[2]],
                       [0, 0, 0, 1]], dtype=np.float32)
    return vox2ras


def nifti_vox2ras(niifile) -> np.ndarray:
    """Get the voxel to RAS transform from NIfTI image.
    Method from the forums: https://neurostars.org/t/get-voxel-to-ras-transformation-from-nifti-file/4549
    
    Args:
        t1: NIfTI image loaded by nibabel.

    Returns:
        np.ndarray: Voxel to RAS transformation.
    """
    nii = nibabel.load(niifile)
    mgh = nibabel.MGHImage(nii.dataobj, nii.affine)
    return mgh.header.get_vox2ras_tkr()


def get_slice_number(img: nibabel.nifti1, zoff: float)->int:
    """Helper function to get the slice number from a NIfTI image that 
    corresponds to a given z-offset (true spatial location in mm).
    
    NOTE: recall the z-off convention comes from the equation of a plane.
            Recall the plane equation is given by ax + by + cz + d = 0.
            The z-offset is the value of d (negative values move you in the superior direction).

    Args:
        img (nibabel.nifti1): The NIfTI image.
        zoff (float): The z-offset in mm.

    Returns:
        int: The slice number.
    """
    bz = img.affine[:,3][:3][-1]  # z-coordinate of the bottom of the image
    return np.rint( (-1*zoff - bz) / img.header.get_zooms()[-1] ).astype(int)


# ----------------------------------------------
# Data wrangling utilities.
# ----------------------------------------------

def niftiPointwiseObservationOp(exnii: str, Vh: dl.FunctionSpace):
    """Assemble a pointwise observation operator for a NIfTI image.

    Args:
        exnii (str): Path to the NIfTI image to be used as a template.
        Vh (dl.FunctionSpace): The dolfin function space.

    Returns:
        The pointwise observation operator object.
    """
    # Get NIfTI image data.
    template = nibabel.load(exnii)
    nx, ny, nz = template.shape          # shape of the NIfTI image
    xyz0 = template.affine[:,3][:3]      # origin of the NIfTI image
    h = template.header.get_zooms()      # voxel size of the NIfTI image

    # create a meshgrid of the observation points (center of the voxels).
    
    xi = np.arange(xyz0[0], xyz0[0] + nx*h[0], h[0])
    yi = np.arange(xyz0[1], xyz0[1] + ny*h[1], h[1])
    zi = np.arange(xyz0[2], xyz0[2] + nz*h[2], h[2])
    
    # todo: do voxels need to be centered (offset by 0.5)?
    # xi = np.arange(xyz0[0] + 0.5*h[0], xyz0[0] + nx*h[0], h[0])
    # yi = np.arange(xyz0[1] + 0.5*h[1], xyz0[1] + ny*h[1], h[1])
    # zi = np.arange(xyz0[2] + 0.5*h[2], xyz0[2] + nz*h[2], h[2])
    
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing='ij')
    
    # stack the meshgrid into an N x 3 array of observation points.
    points = np.hstack([np.reshape(xyz, (xyz.size,1)) for xyz in [xx, yy, zz]])

    # assemble the pointwise observation operator.
    obsOp = assemblePointwiseObservation(Vh, points)
    
    return obsOp


def rasterizeFunction(uh: dl.Function, Vh: dl.FunctionSpace, exnii: str, out: str, obsOp=None) -> None:
    """Write Dolfin function to NIfTI file.

    Args:
        uh (dl.Function): Function to write to NIfTI.
        Vh (dl.FunctionSpace): Function space associated with function.
        exnii (str): Path to NIfTI image to extract header information from.
        out (str): Output filepath for output NIfTI image to be written to.
        
    Returns:
        None (function writes to out).
    """
    
    comm = Vh.mesh().mpi_comm()
    
    if obsOp is None:
        # make the pointwise observation operator.
        obsOp = niftiPointwiseObservationOp(exnii, Vh)
    
    # apply the observation operator to the function to extract the data.
    # NOTE: there is an extremely sublte bug here, you MUST use a dl.PETScVector, not a dl.Vector
    # the local to global map is handled incorrectly. get_local() will only return
    # a zero array. However, the PETScVector will allow you to extract the data
    # from the underlying PETSc object.
    Buh = dl.PETScVector()
    obsOp.init_vector(Buh, 0)
    obsOp.mult(uh.vector(), Buh)

    # set up a scatter to gather the data to rank zero.
    pvec_full = gather_to_zero(Buh.vec())
    
    # data is only on rank zero, so only write it on rank zero.
    if comm.rank == 0:
        # load the template image.
        template = nibabel.load(exnii)
        
        # reshape the data, and write it out to a NIfTI file.
        vals = pvec_full.getArray()
        
        outdata = np.reshape(vals, template.shape)
        outnii = nibabel.Nifti1Image(outdata, affine=template.affine, header=template.header)
        nibabel.save(outnii, out)


def noisyRasterizeFunction(uh: dl.Function, Vh: dl.FunctionSpace, exnii: str, out: str, obsOp=None, noise_std_dev:float=None) -> None:
    """Write Dolfin function to NIfTI file. Apply noise within the tumor region.

    Args:
        uh (dl.Function): Function to write to NIfTI.
        Vh (dl.FunctionSpace): Function space associated with function.
        exnii (str): Path to NIfTI image to extract header information from.
        out (str): Output filepath for output NIfTI image to be written to.
        
    Returns:
        None (function writes to out).
    """
    
    comm = Vh.mesh().mpi_comm()
    
    if obsOp is None:
        # make the pointwise observation operator.
        obsOp = niftiPointwiseObservationOp(exnii, Vh)
    
    # apply the observation operator to the function to extract the data.
    # NOTE: there is an extremely sublte bug here, you MUST use a dl.PETScVector, not a dl.Vector
    # the local to global map is handled incorrectly. get_local() will only return
    # a zero array. However, the PETScVector will allow you to extract the data
    # from the underlying PETSc object.
    Buh = dl.PETScVector()
    obsOp.init_vector(Buh, 0)
    obsOp.mult(uh.vector(), Buh)

    # apply noise to the measurements (if requested)
    if noise_std_dev is not None:
        assert noise_std_dev > 0, "Noise standard deviation must be positive."
        
        
        # apply noise at the observation level so that it's not correlated.

        # apply only with in the domain / ROI
        # u_dummy = dl.project(ufl.conditional(ufl.ge(uh, thresh), dl.Constant(1.0), dl.Constant(0.0)), Vh, solver_type="cg", preconditioner_type="jacobi")
        u_dummy = dl.interpolate(dl.Constant(1.0), Vh)
        support = obsOp * u_dummy.vector()
        
        noise = dl.PETScVector(comm)
        obsOp.init_vector(noise, 0)
        parRandom.normal(noise_std_dev, noise)  # generate noise vector
        
        noise *= support  # apply noise only to the domain / ROI
    
    Buh.vec().axpy(1., noise.vec())

    # set up a scatter to gather the data to rank zero.
    pvec_full = gather_to_zero(Buh.vec())
    
    # data is only on rank zero, so only write it on rank zero.
    if comm.rank == 0:
        # load the template image.
        template = nibabel.load(exnii)
        
        # reshape the data, and write it out to a NIfTI file.
        vals = pvec_full.getArray()
        
        outdata = np.reshape(vals, template.shape)
        outnii = nibabel.Nifti1Image(outdata, affine=template.affine, header=template.header)
        nibabel.save(outnii, out)


def applyFreeSurferShiftRAS(surfdir:str, surf:str, t1:str=None):
    """Apply FreeSurfer C_(RAS) shift to surface.

    Args:
        surfdir (str): Absolute path to the directory with FreeSurfer surface files.
        surf (str): Name of surface file.
        t1 (str, optional): Absolute path to T1 image. If provided, will use the center voxel to RAS shift from the T1 image. Defaults to None.
    """
    # Load in the surface.
    coords, faces, header = nibabel.freesurfer.io.read_geometry(os.path.join(surfdir, surf), read_metadata=True)
    
    if t1 is not None:
        t1 = nibabel.load(t1)
        mgh = nibabel.MGHImage(t1.dataobj, t1.affine)
        cras = nibabel.affines.apply_affine(mgh.header.get_vox2ras(), np.array(mgh.get_fdata().shape) // 2)
        coords += cras
    else:
        # Update coordinates, zero out the shift in the header.
        coords += header['cras']
    
    header['cras'] = np.array([0., 0., 0.])
    
    # Write out the shifted surface.
    nibabel.freesurfer.io.write_geometry(os.path.join(surfdir, f"{surf}.shifted"), coords, faces, volume_info=header)
    
    
def applyROITransformOH(transform: str, fixed: str, moving: str, outdir: str):
    """Resample OncoHabitats ROI to T1 space, by appling the inverse of the provided (ANTsX) transform.
    NOTE: It is assumed that the OncoHabitats registration is provided as as an affine transform from native T1 space to native T1-C space.

    Args:
        transform (str): ANTsX transform .mat file provided by OncoHabitats.
        fixed (str): The fixed image (the image to be deformed to). This is typically the T1.
        moving (str): The moving image (the image to be deformed). This is typically the ROI.
        outdir (str): The output directory for the resampled image.
    """
    
    # Load the transform file and images.
    tf = sitk.ReadTransform(transform)
    fix = sitk.ReadImage(fixed)
    mov = sitk.ReadImage(moving)
    
    # Resample with the inverse transform.
    reg = sitk.Resample(mov, fix, tf.GetInverse(), sitk.sitkNearestNeighbor, 0.0, mov.GetPixelIDValue())

    # Write out the resampled (registered) image.
    sitk.WriteImage(reg, os.path.join(outdir, "ROI_T1.nii"))


def reorientFreeSurferSurf(surfdir:str, surf:str):
    """Reorient FreeSurfer surface from LIA to RAS.
    
    Args:
        surfdir (str): Absolute path to the directory with FreeSurfer surface files.
        surf (str): Name of surface file.
    """
    # Load in the surface.
    coords, faces, header = nibabel.freesurfer.io.read_geometry(os.path.join(surfdir, surf), read_metadata=True)
    
    # invert x/L axis to R orientation
    coords[:, 0] *= -1
    
    # swap y/I and z/A axes to y/A and z/I
    coords[:, [1, 2]] = coords[:, [2, 1]]
    faces[:, [1, 2]] = faces[:, [2, 1]]
    
    # invert y/I axis to A orientation
    coords[:, 1] *= -1
    
    # apply the CRAS shift
    coords += header['cras']
    
    nibabel.freesurfer.io.write_geometry(os.path.join(surfdir, f"{surf}.reoriented"), coords, faces, volume_info=header)
    
    
def computeTumorCellularity(adc:nibabel.nifti1, mask:nibabel.nifti1)->np.ndarray:
    """Compute tumor cellularity from T1 and ADC images.
    
    :code:`WARNING` often the ADC image is in x10^{-6} mm^2/s, so we need to convert to x10^{-3} mm^2/s.
    Please ensure this is appropriate for your dataset.
    
    :code:`WARNING` the two NIfTI images must have the same dimensions.
    
    Args:
        adc (nibabel.nifti1): ADC image (in T1 space).
        mask (nibabel.nifti1): Boolean tumor mask (in T1 space). 1: tumor, 0: healthy.

    Returns:
        nibabel.nifti1: Tumor cellularity.
    """   
    # Constants.
    ADC_W = 3  # x10^{-3} mm^2/s, ADC of free water c.f. Jarrett et. al. 2021
    E3 = 10**3
    E6toE3 = 1/E3
    
    # Unpack NIfTI images.
    adcdata = adc.get_fdata()
    adcdata *= E6toE3  # convert to x10^{-3} mm^2/s scaling
    
    maskdata = mask.get_fdata()
    maskdata[maskdata > 0] = 1  # Make sure the segmentation is binary
    
    assert adcdata.shape == maskdata.shape, "ADC and mask must have the same dimensions."
    
    masked_adc = np.ma.masked_array(adcdata, mask=np.logical_not(maskdata))  # mask out non-tumor voxels
    
    # Compute tumor cellularity.
    ADC_MIN = np.min(masked_adc)  # minimum ADC value in tumor
    PIX = adc.header.get_zooms()[0] * adc.header.get_zooms()[1]  # pixel area in mm^2
    
    cellularity = np.zeros(adc.header.get_data_shape())
    cellularity[np.where(maskdata)] = (ADC_W - (adcdata[np.where(maskdata)] / PIX)) / (ADC_W - (ADC_MIN / PIX))
    cellularity[cellularity < 0] = 0  # set negative values to 0
        
    # Pack results into a NIfTI.
    cellimg = nibabel.Nifti1Image(cellularity, affine=adc.affine, header=adc.header)
    
    return cellimg


def nifti2Function(nii: str, u: dl.Function, Vh: dl.FunctionSpace, zoff: float=None, roi: bool=False) -> None:
    """Write NIfTI voxel data to Dolfin function.

    Args:
        nii (str): Path to NIfTI image.
        u (dl.Function): Function to write tumor to.
        Vh (dl.FunctionSpace): Function space associated with function.
        zoff (float, optional): z-coordinate to slice at if using a 2D mesh. Defaults to None.
        
    Returns:
        None (function writes to u).
    """
    # Get NIfTI image data.    
    image = nibabel.load(nii)
    data = image.get_fdata()
    
    if roi:
        # binary mask
        data[data > 0] = 1
    
    if zoff is not None:
        # 2D-slice, slice the data at the appropriate level.
        slice_number = get_slice_number(image, zoff)
        data = data[:, :, slice_number]
        
        # Higher order interpolation, use hIPPYlib implementation.
        fn = NumpyScalarExpression2D()
        fn.setData(data, *image.header.get_zooms()[:-1])
        fn.setOffset(*image.affine[:,3][:3][:-1])
    else:
        # Higher order interpolation, use hIPPYlib implementation.
        fn = NumpyScalarExpression3D()
        fn.setData(data, *image.header.get_zooms())
        fn.setOffset(*image.affine[:,3][:3])
    
    # Interpolate function.
    fn = dl.interpolate(fn, Vh)
    
    u.vector().zero()  # make sure vector is zero before adding data
    u.vector().axpy(1., fn.vector())
    
    
def vectorNifti2Function(nii: str, u: dl.Function, Vh: dl.FunctionSpace) -> None:
    """Write NIfTI deformation field voxel data to Dolfin function.
    # todo: add support for a sliced meshes (2D).

    Args:
        nii (str): Path to NIfTI image.
        u (dl.Function): Function to write tumor to.
        Vh (dl.FunctionSpace): Function space associated with function.
        
    Returns:
        None (function writes to u).
    """
    image = nibabel.load(nii)
    data = image.get_fdata().squeeze()  # often deformation fields come as (nx, ny, nz, 1, 3)
    
    # Higher order interpolation, use hIPPYlib implementation.
    fn = NumpyVectorExpression3D(Vh.num_sub_spaces())
    
    fn.setData(data, *image.header.get_zooms()[:3])
    fn.setOffset(*image.affine[:,3][:3])
    
    fn = dl.interpolate(fn, Vh)
    u.vector().axpy(1., fn.vector())


def computeCarryingCapacity(nii: str)->float:
    """Compute carrying capacity from voxel volume.

    Args:
        nii (str): Path to NIfTI file.

    Returns:
        float: Carrying capacity [# cells].
    """
    # compute the voxel volume
    nii = nibabel.load(nii)
    voxvol = np.prod(nii.header.get_zooms())
    
    # compute the carrying capacity
    UM_TO_MM = 1E-3
    PACKING_DENSITY = 0.7405    # c.f. Jarrett et. al. 2021
    TUMOR_RADIUS    = 10        # um, c.f. Jarrett et. al. 2021
    TUMOR_RADIUS = TUMOR_RADIUS * UM_TO_MM          # convert to mm
    TUMOR_VOLUME = 4/3 * math.pi * TUMOR_RADIUS**3  # [mm^3]

    return PACKING_DENSITY * voxvol / TUMOR_VOLUME


def makeMisfitTD(pinfo, Vh: dl.FunctionSpace, bc: list, noise_var: float, zoff: float=None, nholdout: int=1, pointwise: bool=False) -> tuple:
    """Make misfits for the tumor growth model.

    Args:
        pinfo (PatientData): PatientData object.
        Vh (dl.FunctionSpace): The (state) function space.
        bc (list): The list of boundary conditions.
        noise_var (float): Noise variance associated with the misfit. (Assumed to be identical for all misfits.)
        zoff (float, optional): z-coordinate to slice at if using a 2D mesh. Defaults to None.
        nholdout (int, optional): How many data points to hold out for prediction. Defaults to 1.

    Returns:
        tuple: The TD misfit object, the imaging timeline.
    """
    MISFITS = []
    
    # use pointwise state observation.
    if pointwise:
        # grab example image to assemble the observation operator (first tumor image)
        exnii = pinfo.get_visit(0).tumor
        obsOp = niftiPointwiseObservationOp(exnii, Vh)
        
        for i, visit in enumerate(pinfo.visits[1:-nholdout]):
            
            # load in the NIfTI image data and flatten (every process will have a copy)
            npdata = nibabel.load(visit.tumor).get_fdata().flatten()
            
            # initialize a PETScVector to store the data (compatible with the observation operator)
            dvec = dl.PETScVector(obsOp.mpi_comm())
            obsOp.init_vector(dvec, 0)
            
            numpy2Vec(dvec, npdata)
            
            misfit = DiscreteStateObservation(obsOp, dvec, noise_variance=noise_var)
            MISFITS.append(misfit)
    else:
        # use continuous state observation.
        uhelp = dl.Function(Vh)
        for i, visit in enumerate(pinfo.visits[1:-nholdout]):
            uhelp.vector().zero()
            nifti2Function(visit.tumor, uhelp, Vh, zoff)
            misfit = ContinuousStateObservation(Vh, dl.dx, bc, uhelp.vector(), noise_variance=noise_var)
            MISFITS.append(misfit)
    
    misfit_obj = MisfitTD(MISFITS, pinfo.visit_days[:-nholdout])
    
    return misfit_obj


def applyGaussianBlur(img:  nibabel.nifti1, sigma: float=1., clip: bool=True) -> nibabel.nifti1:
    """Apply Gaussian Blur to NIfTI image data.

    Args:
        img (nibabel.nifti1): Original NIfTI image.
        sigma (float, optional): Standard deviation of Gaussian blur. Defaults to 1.0 (unitless, voxel space).
        clip (bool, optional): Whether or not to clip blur to be in [0, 1]. Useful when dealing with tumor cellularity data. Defaults to True.

    Returns:
        nibabel.nifti1: NIfTI image with Gaussian blur applied.
    """
    imgdata = img.get_fdata()
    filtered = scipy.ndimage.gaussian_filter(imgdata, sigma=sigma)
    
    if clip:
        np.clip(filtered, 0, 1, out=filtered)
    
    out = nibabel.Nifti1Image(filtered, affine=img.affine, header=img.header)
    return out


# ----------------------------------------------
# Deprecated utilities.
# ----------------------------------------------

def markTumorSegmentation(niifile: str, mesh: dl.Mesh, bool_mask: bool=True)->dl.MeshFunction:
    """Create mesh function with markers for tumor segmentation.

    Args:
        niifile (str): Path to the tumor segmentation NIfTI file.
        mesh (dl.Mesh): Mesh to create MeshFunction on.
        bool_mask (bool, optional): Whether or not to mask all tumor or keep segmentation values. Defaults to True.

    Returns:
        markers (dl.MeshFunction): Mesh function with markers for tumor segmentation.
    """
    # Load in the NIfTI image data.
    image = nibabel.load(niifile)
    data = image.get_fdata()
    
    n = mesh.topology().dim()
       
    if bool_mask:
        # Mesh function to store segmentation data.
        markers = dl.MeshFunction("bool", mesh, n, False)
        
        # Build voxel to RAS map.
        vox2ras = get_vox2ras(image)

        # RAS to voxel map.
        ras2vox = np.linalg.inv(vox2ras)

        # Loop through mesh vertex coordinates and grab the closest thing out of the voxel image?
        for vertex in dl.vertices(mesh):
            c = vertex.index()
            
            # Extract the RAS coordinates of the vertex midpoint
            xyz = vertex.midpoint()[:]
            
            # Convert to voxel space.
            ijk = apply_affine(ras2vox, xyz)
            
            # Round off to nearest integers to find voxel indices.
            i, j, k = np.rint(ijk).astype("int")
            
            # Insert image data into the function. If data can't be read, then assume no tumor.
            try:
                if data[i, j, k] != 0:
                    # Tag the region as tumor with a memorable value
                    markers.array()[c] = True
            except:
                markers.array()[c] = False
    else:
        # Mesh function to store segmentation data.
        markers = dl.MeshFunction("size_t", mesh, n)
        
        # Extract information about the data.
        ds = image.header.get_zooms()
        ns = np.array(image.shape[:3]) * ds / 2.0
        numpy2MeshFunction(mesh, h=ds, offsets=-ns, data=data, mfun=markers)

    return markers
    

def applyInitialCondition(niifile: str, 
                          mesh: dl.Mesh,
                          u: dl.Function,
                          V: dl.FunctionSpace,
                          useIntermediate: bool=False,
                          enhanceval: float=0.8,
                          nonenhanceval: float=0.16,
                          use_vox2ras: bool=False) -> None:
    """Tumor initial condition as specified from a NIfTI image.

    Args:
        niifile (str): Path to NIfTI image containing tumor segmentations.
        mesh (dl.Mesh): Computational mesh.
        u (dl.Function): Function to write tumor to.
        V (dl.FunctionSpace): Function space associated with function.
        useIntermediate (bool, optional): Whether or not to make intermediate image for generating voxel to RAS map. Defaults to False.
        enhanceval (float, optional): Enhancing tumor density. Defaults to 0.8.
        nonenhanceval (float, optional): Non-enhancing tumor density. Defaults to 0.16.
        degree (int, optional): Degree of polynomial to use for interpolation. Defaults to 1.
        
    Returns:
        None (function writes to u).
    """
    image = nibabel.load(niifile)
    data = image.get_fdata()
    
    # Convert segmentation label to tumor density.
    data[data == 1] = nonenhanceval
    data[data == 2] = nonenhanceval
    data[data == 3] = nonenhanceval
    data[data == 4] = enhanceval
    
    if use_vox2ras:
        # Build voxel to RAS map.
        if useIntermediate:
            # Option 1: go through intermediate image.
            # mgh = nibabel.MGHImage(image.dataobj, image.affine)
            # vox2ras = mgh.header.get_vox2ras_tkr()
            # TODO: known bug, cannot rotate data appropriately to work with the voxel to RAS map.
            raise NotImplementedError
        else:
            # Option 2: use function to rebuild vox2ras map
            
            vox2ras = get_vox2ras(image)

        # RAS to voxel map.
        ras2vox = np.linalg.inv(vox2ras)

        # Vertex to degree of freedom map.
        v2dof = dl.vertex_to_dof_map(V)

        # Loop through mesh vertex coordinates and grab the closest thing out of the voxel image?
        for vertex in dl.vertices(mesh):
            c = vertex.index()
            
            # Extract the RAS coordinates of the vertex midpoint
            xyz = vertex.midpoint()[:]
            
            # Convert to voxel space.
            ijk = apply_affine(ras2vox, xyz)
            
            # Round off to nearest integers to find voxel indices.
            i, j, k = np.rint(ijk).astype("int")
            
            # Insert image data into the function. If data can't be read, then assume no tumor.
            try:
                u.vector()[v2dof[c]] = data[i,j,k]
            except:
                u.vector()[v2dof[c]] = 0.
    else:
        # Higher order interpolation, use hIPPYlib implementation.
        fn = NumpyScalarExpression3D()
        
        # Get NIfTI image data.
        ds = image.header.get_zooms()
        ns = np.array(image.shape[:3]) * ds / 2.0
        fn.setData(data, *ds)
        fn.setOffset(*-ns)
        
        # Interpolate function.
        fn = dl.interpolate(fn, V)
        u.assign(fn)


def legacy_computeImagingTimeline(dir:str)->tuple:
    """Compute the imaging timeline from the directory structure / data.

    Args:
        dir (str): path to the patient directory.

    Returns:
        np.array: imaging days.
    """
    # Read the dates.
    dates = [f.name for f in os.scandir(dir) if f.is_dir()]
    dates.sort()  # just to be sure
    # Convert to datetime objects.
    dtimes = [datetime.strptime(date, "%Y_%m_%d") for date in dates]
    # Compute the days since the first image.
    times = np.array([(d - dtimes[0]).days for d in dtimes]).astype(float)
    
    return dates, times


def legacy_makeMisfitTD(pdir: str, Vh: dl.FunctionSpace, bc: list, noise_var: float, zoff: float=None, nholdout: int=1)->tuple:
    """Make misfits for the tumor growth model.

    Args:
        pdir (str): Path to directory contatining patient data.
        Vh (dl.FunctionSpace): The function space.
        bc (list): The list of boundary conditions.
        noise_var (float): Noise variance associated with the misfit. (Assumed to be identical for all misfits.)
        zoff (float, optional): z-coordinate to slice at if using a 2D mesh. Defaults to None.
        nholdout (int, optional): How many data points to hold out for prediction. Defaults to 1.

    Returns:
        tuple: The TD misfit object, the imaging timeline.
    """
    MISFITS = []
    
    u = dl.Function(Vh)
    
    imgdates, times = legacy_computeImagingTimeline(pdir)
    BASELINE_PATH = os.path.join(pdir, imgdates[0])
    
    for i, date in enumerate(imgdates[:-nholdout]):
        DENSITYFPATH = os.path.join(BASELINE_PATH, "reg", date, "celldensity_FS.nii")
        nifti2Function(DENSITYFPATH, u, Vh, zoff)
        misfit = ContinuousStateObservation(Vh, dl.dx, bc, u.vector(), noise_variance=noise_var)
        MISFITS.append(misfit)

    misfit_obj = MisfitTD(MISFITS, times[:-nholdout])

    return misfit_obj, times

def legacy_loadIC(pdir: str, Vh: dl.FunctionSpace, blur: bool=False, zoff: float=None, phantom: bool=False)->dl.Function:
    """Load the tumor initial condition from the first imaging date.

    Args:
        cfg: Configuration dictionary.
        Vh (dl.FunctionSpace): Function space for state.
        blur (bool, optional): Whether or not to use the blurred initial condition. Defaults to False (unblurred).
        zoff (float, optional): z-coordinate to slice at if using a 2D mesh. Defaults to None.

    Returns:
        dl.Function: The initial condition.
    """
    u = dl.Function(Vh)
    imgdates, _ = legacy_computeImagingTimeline(pdir)
    if blur:
        if phantom:
            DENSITYFPATH = os.path.join(pdir, imgdates[0], "reg", imgdates[0], "celldensity_phantom_blur.nii")
        else:
            DENSITYFPATH = os.path.join(pdir, imgdates[0], "reg", imgdates[0], "celldensity_FS_blur.nii")
    else:
        if phantom:
            DENSITYFPATH = os.path.join(pdir, imgdates[0], "reg", imgdates[0], "celldensity_phantom.nii")
        else:
            DENSITYFPATH = os.path.join(pdir, imgdates[0], "reg", imgdates[0], "celldensity_FS.nii")
    
    nifti2Function(DENSITYFPATH, u, Vh, zoff)
    
    return u
