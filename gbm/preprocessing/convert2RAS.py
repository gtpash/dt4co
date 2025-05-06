import os
import argparse
from pathlib import Path

def main(args) -> None:
    # ---------------------------------------------------------------
    # Ensure RAS orientation.
    # ---------------------------------------------------------------
    for i, nifti in enumerate(Path(args.dir).glob("*.nii")):
        os.system(f"mri_convert {nifti} {nifti} --out_orientation 'RAS'")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NIfTI files to RAS orientation.")
    parser.add_argument("--dir", type=str, help="Path to the directory containing the NIfTI files.")
    args = parser.parse_args()
    
    main(args)
