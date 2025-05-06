###############################################################################
# 
# Small utility to concatenate the first visit ROIs for KUL_VBG.
# 
# Usage: python3 roi_concat.py --roie /path/to/roi/enhancing --roin /path/to/roi/nonenhancing --out /path/to/store
# 
###############################################################################

import argparse
import nibabel

def main(args) -> None:
    # load the ROIs.
    roi_e = nibabel.load(args.roie)
    roi_ne = nibabel.load(args.roin)
    
    # combine the data.
    comb_data = roi_e.get_fdata() + roi_ne.get_fdata()
    comb_data[comb_data > 0] = 1  # ensure that the image is binary.
    
    outnii = nibabel.Nifti1Image(comb_data, affine=roi_e.affine, header=roi_e.header)
    
    # write out the image.
    nibabel.save(outnii, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate ROIs for KUL_VBG.")
    
    parser.add_argument("--roie", type=str, help="Path to enhancing tumor ROI.")
    parser.add_argument("--roin", type=str, help="Path to non-enhancing tumor ROI.")
    parser.add_argument("--out", type=str, help="File to save output to.")

    args = parser.parse_args()
    main(args)
