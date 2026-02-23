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
        expected = np.array(
            [18.0, 19.0, 20.0, 21.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 32.0, 33.0, 34.0, 35.0, 36.0, 39.0, 40.0, 41.0, 42.0, 43.0, 46.0, 47.0, 48.0, 49.0, 50.0, 53.0, 54.0, 55.0, 56.0, 57.0]
        )

        # check the radiotherapy days and doses.
        assert np.array_equal(self.datamodel.radio_days, expected), "The radiotherapy days are not as expected."
        assert np.array_equal(self.datamodel.radio_doses, 2.0 * np.ones_like(expected)), "The radiotherapy doses are not as expected."

    def test_ChemotherapyTreatment(self):
        expected = np.array(
            [
                17.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
                25.0,
                26.0,
                27.0,
                28.0,
                29.0,
                30.0,
                31.0,
                32.0,
                33.0,
                34.0,
                35.0,
                36.0,
                37.0,
                38.0,
                39.0,
                40.0,
                41.0,
                42.0,
                43.0,
                44.0,
                45.0,
                46.0,
                47.0,
                48.0,
                49.0,
                50.0,
                51.0,
                52.0,
                53.0,
                54.0,
                55.0,
                56.0,
                57.0,
                58.0,
                59.0,
                60.0,
                61.0,
                62.0,
                63.0,
                64.0,
                65.0,
                66.0,
                67.0,
                68.0,
                69.0,
                70.0,
                83.0,
                84.0,
                85.0,
                86.0,
                87.0,
                111.0,
                112.0,
                113.0,
                114.0,
                115.0,
                139.0,
                140.0,
                141.0,
                142.0,
                143.0,
                167.0,
                168.0,
                169.0,
                170.0,
                171.0,
            ]
        )

        # check the chemotherapy days.
        assert np.array_equal(self.datamodel.chemo_days, expected), "The chemotherapy days are not as expected."
        assert np.array_equal(self.datamodel.chemo_effects, 0.82 * np.ones_like(expected)), "The chemotherapy effects are not as expected."


if __name__ == "__main__":
    unittest.main()
