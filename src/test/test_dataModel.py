import unittest
import os

from datetime import datetime
from pathlib import Path
import numpy as np

import sys
sys.path.append("../")
from dt4co.dataModel import PatientData

class TestDataModel(unittest.TestCase):
    def setUp(self):
        self.ex_dir = "test"
        self.datamodel = PatientData(os.path.join("data", "ex_patient_info.json"), self.ex_dir)
    
    def test_get_timeline(self):
        # check the timeline method.
        expected = [datetime(1999, 4, 10).date(), datetime(1999, 7, 2).date(), datetime(1999, 8, 20).date(), datetime(1999, 10, 9).date(), datetime(1999, 11, 15).date()]
        self.assertListEqual(expected, self.datamodel.get_timeline(), "Timeline is not as expected.")
        
        assert len(self.datamodel.get_timeline()) == self.datamodel.get_num_visits(), "The number of visits is not as expected."
    
    def test_VisitData(self):
        expected_time = datetime(1999, 4, 10).date()
        expected_tumor = Path(os.path.join(self.ex_dir, "tumor_blur_v1_fs.nii"))
        expected_roi = Path(os.path.join(self.ex_dir, "ROI_v1_fs.nii"))
        
        assert self.datamodel.get_visit(0).time == expected_time, "Visit time is not as expected."
        assert self.datamodel.get_visit(0).tumor == expected_tumor, "Visit tumor is not as expected."
        assert self.datamodel.get_visit(0).roi == expected_roi, "Visit ROI is not as expected."
        
    def test_get_visit(self):
        assert self.datamodel.get_visit(0).time == self.datamodel.visits[0].time, "The correct time is not being returned."
        assert self.datamodel.get_visit(0).tumor == self.datamodel.visits[0].tumor, "The correct tumor is not being returned."
        assert self.datamodel.get_visit(0).roi == self.datamodel.visits[0].roi, "The correct ROI is not being returned."
    
    def test_RadiotherapyTreatment(self):
        expected = np.array([18., 19., 20., 21., 22., 25., 26., 27., 28., 29., 32., 33., 34., 35., 36., 39., 40., 41., 42., 43., 46., 47., 48., 49., 50., 53., 54., 55., 56., 57.])
        
        # check the radiotherapy days and doses.
        assert np.array_equal(self.datamodel.radio_days, expected), "The radiotherapy days are not as expected."
        assert np.array_equal(self.datamodel.radio_doses, 2.*np.ones_like(expected)), "The radiotherapy doses are not as expected."
    
    def test_ChemotherapyTreatment(self):
        expected = np.array([ 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 83., 84., 85., 86., 87., 111., 112., 113., 114., 115., 139., 140., 141., 142., 143., 167., 168., 169., 170., 171.])
        
        # check the chemotherapy days.
        assert np.array_equal(self.datamodel.chemo_days, expected), "The chemotherapy days are not as expected."
        assert np.array_equal(self.datamodel.chemo_effects, 0.82*np.ones_like(expected)), "The chemotherapy effects are not as expected."


if __name__ == '__main__':
    unittest.main()

