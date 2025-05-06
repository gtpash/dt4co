import dolfin as dl
import ufl
import numpy as np
import nibabel

def computeDice(x: dl.Function, y: dl.Function, threshold: float=0.5) -> float:
    """Compute the Dice-Sorensen coefficient between two functions x and y.
    The Dice-Sorensen coefficient is defined as:
    DICE = 2 * |x cap y| / (|x| + |y|)
    where |x| is the volume of the level set x
    |y| is the volume of the level set y
    |x cap y| is the volume of the intersection of x and y
    Note that the Dice coefficient is symmetric in x and y and takes values in [0, 1].
    The Dice coefficient is 1 if and only if x = y.
    The Dice coefficient is 0 if and only if x and y do not intersect.
    
    NOTE: Dice(x, x) may not be 1.0 due to interpolation errors.

    Args:
        x (dl.Function): First function.
        y (dl.Function): Second function.
        threshold (float, optional): Threshold value for state to compute level set. Defaults to 0.5.

    Returns:
        float: The Dice-Sorensen coefficient.
    """
    
    # threshold the functions.
    xth = ufl.conditional(ufl.ge(x, threshold), dl.Constant(1.0), dl.Constant(0.0))
    yth = ufl.conditional(ufl.ge(y, threshold), dl.Constant(1.0), dl.Constant(0.0))
    
    # compute the volume of the level set x
    xvol = dl.assemble(xth * ufl.dx)
    yvol = dl.assemble(yth * ufl.dx)
    
    # compute the volume of the intersection of x and y.
    volxy = dl.assemble(xth * yth * ufl.dx)
    
    # compute the DICE coefficient.
    dice = 2 * volxy / (xvol + yvol)
    
    return dice


def computeNTV(u: dl.Function, threshold: float=0.5) -> float:
    """Compute the normalized tumor volume of a function u.
    NTV := ( 1 / |Omega| ) * int_{Omega} u dx
    where |Omega| is the volume of the domain Omega.
    The NTV is a measure of the volume of the tumor in the domain Omega.

    Args:
        u (dl.Function): Function containing state.
        threshold (float, optional): Threshold value for state to compute the level set. Defaults to 0.5.

    Returns:
        float: The normalized tumor volume.
    """
    
    uth = ufl.conditional(ufl.ge(u, threshold), dl.Constant(1.0), dl.Constant(0.0))
    tumor = dl.assemble(uth * ufl.dx)
    
    dx = ufl.dx(u.function_space().mesh())  # measure of the domain
    
    vol = dl.assemble(dl.Constant(1.) * dx)  # of the domain
    
    return tumor / vol


def computeTTV(u: dl.Function, threshold: float=0.5) -> float:
    """Compute the total tumor volume of a function u.
    TTV := int_{Omega} u dx
    where |Omega| is the volume of the domain Omega.
    The TTV is a measure of the volume of the tumor in the domain Omega.

    Args:
        u (dl.Function): Function containing state.
        threshold (float, optional): Threshold value for state to compute the level set. Defaults to 0.5.

    Returns:
        float: The total tumor volume.
    """
    
    uth = ufl.conditional(ufl.ge(u, threshold), dl.Constant(1.0), dl.Constant(0.0))
    ttv = dl.assemble(uth * ufl.dx)
    
    return ttv


def computeTTC(u: dl.Function, carry_cap: float, threshold: float=None) -> float:
    """Compute the total tumor cellularity.

    Args:
        u (dl.Function): The tumor state.
        carry_cap (float): Carrying capacity [cells/mm^3]
        threshold (float, optional): Threshold value for state to compute level set. Defaults to None.

    Returns:
        float: total tumor cellularity.
    """
    
    if threshold is not None:
        uth = ufl.conditional(ufl.ge(u, threshold), dl.Constant(1.0), dl.Constant(0.0))
        ttc = dl.assemble(dl.Constant(carry_cap) * uth * u * ufl.dx)
    else:
        ttc = dl.assemble(dl.Constant(carry_cap) * u * ufl.dx)

    return ttc
    
    
def compute_ccc(x: np.ndarray, y: np.ndarray, bias: bool=True, use_pearson: bool=False) -> float:
    """Compute the concordance correlation coefficient between two arrays x and y.

    Args:
        x (np.ndarray): First vector.
        y (np.ndarray): Second vector.
        bias (bool, optional): Bias correction. Defaults to True.
        use_pearson (bool, optional): Use Pearson correlation coefficient. Defaults to False.

    Returns:
        float: The concordance correlation coefficient.
    """
    
    if use_pearson:
        cor = np.corrcoef(x, y)[0][1]  # Pearson correlation coefficient
        
        # compute means.
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # compute variances.
        var_x = np.var(x)
        var_y = np.var(y)
        
        # compute standard deviations.
        sd_x = np.std(x)
        sd_y = np.std(y)
        
        # compute the CCC.
        numerator = 2 * cor * sd_x * sd_y
        denominator = var_x + var_y + (mean_x - mean_y)**2
        ccc = numerator / denominator
    else:
        # compute the covariance.
        var_x, cov_xy, cov_xy, var_y = np.cov(x, y, bias=bias).flat
        
        # compute means.
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # compute the CCC.
        numerator = 2 * cov_xy
        denominator = var_x + var_y + (mean_x - mean_y)**2
        ccc = numerator / denominator
    
    return ccc


def computeVoxelCCC(modelnii: str, truenii: str, roinii: str, bias: bool=True, use_pearson: bool=False) -> float:
    """Compute the voxel-wise concordance correlation coefficient between two NIfTI files.

    Args:
        modelnii (str): Path to x NIfTI file.
        truenii (str): Path to y NIfTI file.
        roinii (str): Path to the ROI NIfTI file.

    Returns:
        np.array: Voxel-wise concordance correlation coefficient.
    """
    
    # load in the model and data NIfTI files, check compatibility.
    x = nibabel.load(modelnii)
    y = nibabel.load(truenii)
    
    assert x.shape == y.shape, "Model and data must have the same shape."
    assert x.header.get_zooms() == y.header.get_zooms(), "Model and data must have the same voxel size."
    assert np.array_equal(x.affine, y.affine), "Model and data must have the same affine matrix."
    
    # load in the ROI NIfTI file, check compatibility.
    roi = nibabel.load(roinii)
    assert x.shape == roi.shape, "ROI does not have the same shape as model and data."
    assert x.header.get_zooms() == roi.header.get_zooms(), "ROI does not have the same voxel size as model and data."
    assert np.array_equal(x.affine, roi.affine), "ROI does not have the same affine matrix as model and data."
    
    # extract the data from the NIfTI files.
    x_data = x.get_fdata()
    y_data = y.get_fdata()
    
    # mask the data with the ROI.
    roi_mask = roi.get_fdata()
    roi_mask[roi_mask > 0] = 1  # Make sure the segmentation is binary
    
    # mask the data with the ROI.
    x_roi = np.ma.masked_array(x_data, mask=np.logical_not(roi_mask))  # mask out non-tumor voxels
    y_roi = np.ma.masked_array(y_data, mask=np.logical_not(roi_mask))  # mask out non-tumor voxels
    
    # reshape the data to 1D arrays.
    x_vox = np.reshape(x_roi, -1)
    y_vox = np.reshape(y_roi, -1)
    
    # compute the CCC.
    ccc = compute_ccc(x_vox, y_vox, bias, use_pearson)
    
    return ccc


def compute_dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Dice coefficient between y_true and y_pred.

    Args:
        y_true (np.ndarray): Ground truth mask
        y_pred (np.ndarray): Predicted mask

    Returns:
        float: The computed Dice coefficient.
    """
    
    # Flatten the tensors
    y_true_f = np.reshape(y_true, -1)
    y_pred_f = np.reshape(y_pred, -1)
    
    # Compute the intersection and the sum of the two masks
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    
    # Compute Dice coefficient
    dice_coeff = (2. * intersection) / union
    
    return dice_coeff


def computeVoxelDice(xnii: str, ynii: str, threshold: float=0.5) -> float:
    """Compute voxel-wise Dice coefficient between two NIfTI files.

    Args:
        xnii (str): Path to x NIfTI file.
        ynii (str): Path to y NIfTI file.
        threshold (float, optional): Value to threshold on. Defaults to 0.5. Useful for tumor cellularity maps.

    Returns:
        float: Dice coefficient.
    """
    # load in the model and data NIfTI files, check compatibility.
    x = nibabel.load(xnii)
    y = nibabel.load(ynii)
    
    assert x.shape == y.shape, "Model and data must have the same shape."
    assert x.header.get_zooms() == y.header.get_zooms(), "Model and data must have the same voxel size."
    assert np.array_equal(x.affine, y.affine), "Model and data must have the same affine matrix."
    
    # extract the data from the NIfTI files.
    x_data = x.get_fdata()
    y_data = y.get_fdata()
    
    xth = np.where(x_data > threshold, 1, 0)
    yth = np.where(y_data > threshold, 1, 0)
    
    dice = compute_dice(xth, yth)
    return dice
