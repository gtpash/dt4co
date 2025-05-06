import unittest
import os
import sys

import dolfin as dl

sys.path.append("../")
from dt4co.utils.data_utils import nifti2Function, rasterizeFunction
from dt4co.qoi import computeVoxelDice, computeDice

class TestObservationOperator(unittest.TestCase):
    def setUp(self):
        """Set up MPI communicator, paths to data.
        """
        self.comm = dl.MPI.comm_world
        self.mesh_fpath = os.path.join("data", "mesh.xdmf")
        self.t1 = os.path.join("data", "T1_pre.nii")
        self.roi = os.path.join("data", "roi.nii")
        self.outdir = os.path.join("data", "test")
        
        self.xnii = os.path.join("data", "x_raster.nii")
        self.ynii = os.path.join("data", "y_raster.nii")
        self.znii = os.path.join("data", "z_raster.nii")
        
        os.makedirs(self.outdir, exist_ok=True)
        
        self.mesh = dl.Mesh()
        with dl.XDMFFile(self.comm, self.mesh_fpath) as fid:
            fid.read(self.mesh)
            
        self.Vh = dl.FunctionSpace(self.mesh, "CG", 1)
    
    
    def test_Bstar(self):
        """Test the adjoint of the observation operator, B^* : \mathbb{R}^d -> \mathcal{U}.
        by reading back the unit directions from the NIfTI files.
        """
        # interpolate the unit directions
        xfun = dl.interpolate(dl.Expression("x[0]", degree=2), self.Vh)
        yfun = dl.interpolate(dl.Expression("x[1]", degree=2), self.Vh)
        zfun = dl.interpolate(dl.Expression("x[2]", degree=2), self.Vh)
        
        # read back the unit directions from the NIfTI files.
        xrbfun = dl.Function(self.Vh)
        yrbfun = dl.Function(self.Vh)
        zrbfun = dl.Function(self.Vh)
    
        nifti2Function(self.xnii, xrbfun, self.Vh, None, False)
        nifti2Function(self.ynii, yrbfun, self.Vh, None, False)
        nifti2Function(self.znii, zrbfun, self.Vh, None, False)
        
        # compare the unit directions.
        res = dl.Function(self.Vh)
        res.vector().zero()
        res.vector().axpy(1., xrbfun.vector())
        res.vector().axpy(-1., xfun.vector())
        assert res.vector().norm("linf") < 1e-6, "Unit direction x is not the same."
        
        res.vector().zero()
        res.vector().axpy(1., yrbfun.vector())
        res.vector().axpy(-1., yfun.vector())        
        assert res.vector().norm("linf") < 1e-6, "Unit direction y is not the same."
        
        res.vector().zero()
        res.vector().axpy(1., zrbfun.vector())
        res.vector().axpy(-1., zfun.vector())
        
        assert res.vector().norm("linf") < 1e-2, "Unit direction z is not the same."
        
        
    def test_BBstar(self):
        """Test B(B^*(u)) = u for the ROI.
        """
        uroi = dl.Function(self.Vh)
        nifti2Function(self.roi, uroi, self.Vh, None, False)
        rasterizeFunction(uroi, self.Vh, exnii=self.t1, out=os.path.join(self.outdir, "test_raster_roi.nii"))
        vdice = computeVoxelDice(os.path.join(self.outdir, "test_raster_roi.nii"), self.roi, threshold=0.2)
        
        assert vdice > 0.8, f"B(B^*) too far from identity, Dice = {vdice}"
    
    
    def test_BstarBBstar(self):
        """Test B^*(B(B^*(u))) = B^*(u) for the ROI.
        """
        # project ROI onto mesh
        uroi_orig = dl.Function(self.Vh)
        nifti2Function(self.roi, uroi_orig, self.Vh, None, False)
        
        # write out the ROI from mesh to a NIfTI file
        rasterizeFunction(uroi_orig, self.Vh, exnii=self.roi, out=os.path.join(self.outdir, "test_raster_roi.nii"))
        
        # read back the NIfTI that was just written
        uroi_rb = dl.Function(self.Vh)
        nifti2Function(os.path.join(self.outdir, "test_raster_roi.nii"), uroi_rb, self.Vh, None, False)
        
        # check for identity
        udice = computeDice(uroi_orig, uroi_rb, threshold=0.2)
        
        assert udice > 0.9, f"B^*(B(B^*(u))) too far from identity, Dice = {udice}"
            

if __name__ == '__main__':
    unittest.main()
