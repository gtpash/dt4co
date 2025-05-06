import os
import argparse
import SimpleITK as sitk

def main(args):
    """
    It is assumed that the OncoHabitats registration is provided as as an affine transform from native T1 space to native T1-C space.
    """
    
    # Load the transform file and images.
    tf = sitk.ReadTransform(args.transform)
    fix = sitk.ReadImage(args.fixed)
    mov = sitk.ReadImage(args.moving)
    
    # Resample with the inverse transform.
    reg = sitk.Resample(mov, fix, tf.GetInverse(), sitk.sitkNearestNeighbor, 0.0, mov.GetPixelIDValue())

    # Write out the resampled (registered) image.
    sitk.WriteImage(reg, os.path.join(args.outdir, "Segmentation_T1.nii"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply (inverse) OncoHabitats transform to move segmentation to native T1 space. Ouput file is: Segmentation_T1.nii")

    parser.add_argument("-tf", "--transform", type=str, help="Absolute path to the transform file.")
    parser.add_argument("-m", "--moving", type=str, help="Absolute path to the moving image (the image to be deformed).")
    parser.add_argument("-f", "--fixed", type=str, help="Absolute path to the fixed image (the image to be deformed to).")
    parser.add_argument("-o", "--outdir", type=str, help="Absolute path to the output directory.")
    
    args = parser.parse_args()
    main(args)
