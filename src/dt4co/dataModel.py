###############################################################################
# 
# This file defines the data model for modeling high grade gliomas.
#  
# NOTE
#   - Visits must be ordered by the date of the visit in the JSON file defining the patient data.
# 
###############################################################################

from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, date
from typing import List, Tuple, Dict, TypeAlias
import numpy as np


# Type aliases
TreatmentTime: TypeAlias = date
RadiotherapyProtocol: TypeAlias = Dict[TreatmentTime, float]

def days_since_first(date: datetime, first: datetime) -> float:
    return float((date - first).days)

# ------------------------------
# Data classes
# ------------------------------
@dataclass
class VisitData:
    """Class to hold data for a single visit.
    """
    time: date        # date of the visit
    tumor: Path       # path to the (blurred) tumor density NIfTI file
    roi: Path         # path to the region of interest NIfTI file


@dataclass
class RadiotherapyTreatment:
    """Class to hold information for a single radiotherapy visit.
    """
    time: TreatmentTime     # date of the radiotherapy visits
    dose: float             # dosage of applied radiotherapy [Gy]


@dataclass
class RadiotherapySpecification:
    tx_visits: List[RadiotherapyTreatment]

    @property
    def protocol(self) -> RadiotherapyProtocol:
        return {tx.time: tx.dose for tx in self.tx_visits}


@dataclass
class ChemotherapyTreatment:
    """Class to hold information for a single chemotherapy visit.
    """
    time: TreatmentTime     # date of the chemotherapy visits
    sf: float               # surviving fraction for the chemotherapy dosage
    # todo: add more realistic model for chemotherapy effects


@dataclass
class ChemotherapySpecification:
    tx_visits: List[ChemotherapyTreatment]

    @property
    def protocol(self) -> Dict[TreatmentTime, float]:
        return {tx.time: tx.sf for tx in self.tx_visits}


# ------------------------------
# For the patient.
# ------------------------------
class PatientData:
    """Class to hold data for a single patient.
    """
    def __init__(self, info: Path, data_dir: Path) -> None:
        """Constructor for PatientData
        # todo: add model validation (with pydantic?)

        Args:
            info (Path): Path to JSON file containing patient information.
            data_dir (Path): Path to the directory containing the patient data.
        """
        self.data_dir = data_dir
        
        # read in the patient information, and unpack the visits
        with open(info) as f:
            self.info = json.load(f)
        
        self.visits = self._unpack_visits(self.info)
        self.radio_plan = self._unpack_radiotherapy(self.info)
        self.chemo_plan = self._unpack_chemotherapy(self.info)
        
        
    def _unpack_visits(self, info: dict) -> List[VisitData]:
        """Unpack the visits from the JSON file.
        """
        visits = []
        for visit in info["visits"]:
            visits.append(self._unpack_visit(visit))
        
        return visits
    
        
    def _unpack_visit(self, visit) -> VisitData:
        date_str = visit["time"].split("T")[0]  # split on the "time" (always midnight)
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")  # cast to datetime object
        date_obj = date_obj.date()
        
        # tumor = self._replace_image_dir_variable(visit["tumor_fs"])
        tumor = self._replace_image_dir_variable(visit["tumor_blur_fs"])
        roi = self._replace_image_dir_variable(visit["roi_fs"])
        
        return VisitData(date_obj, tumor, roi)
    
    
    def _replace_image_dir_variable(self, path: Path) -> Path:
        return Path(str(path).replace("{$image_dir}", str(self.data_dir)))
    
    
    def get_timeline(self) -> Tuple[List[datetime]]:
        """Get the dates of the visits.
        """
        dates = []
        for visit in self.visits:
            dates.append(visit.time)
        
        return dates
    
    @property
    def visit_days(self) -> np.array:
        """Compute the number of days since the first visit.
        """
        
        dates = self.get_timeline()       
        return np.array([(d - dates[0]).days for d in dates]).astype(float)
    
    
    def get_visit(self, i: int) -> VisitData:
        """Get the data for the i-th visit.
        """
        return self.visits[i]
    
    
    def get_num_visits(self) -> int:
        """Get the number of visits.
        """
        return len(self.visits)
    
    
    def print_timeline(self) -> None:
        """Print the timeline of the visits.
        """
        dates = self.get_timeline()
        for i, date in enumerate(dates):
            print(f"Visit {i+1}: {date}")
    

    # ------------------------------
    # Radiotherapy
    # ------------------------------

    def _unpack_rt_visit(self, rt_visit) -> RadiotherapyTreatment:
        date_str = rt_visit["time"].split("T")[0]  # split on the "time" (always midnight)
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")  # cast to datetime object
        date_obj = date_obj.date()
        
        dose = rt_visit['dose']
        
        return RadiotherapyTreatment(date_obj, dose)
    
    
    def _unpack_radiotherapy(self, info: dict) -> RadiotherapySpecification:
        """Unpack the visits from the JSON file.
        """
        rt_visits = []
        for rt_visit in info["radiotherapy"]:
            rt_visits.append(self._unpack_rt_visit(rt_visit))
        
        return RadiotherapySpecification(rt_visits)
    
    
    def get_radiotherapy_protocol(self) -> dict:
        """Get the Radiotherapy protocol.

        Returns:
            Dictionary of the radiotherapy protocol.
        """
        first = self.get_timeline()[0]
        return {days_since_first(r.time, first): r.dose for r in self.radio_plan.tx_visits}
    
    
    @property
    def radio_days(self) -> List[float]:
        """Get the number of days since the first visit for each radiotherapy visit.
        """
        first = self.get_timeline()[0]
        return np.array([days_since_first(r.time, first) for r in self.radio_plan.tx_visits])
    
    
    @property
    def radio_doses(self) -> List[float]:
        """Get the doses for each radiotherapy visit.
        """
        return np.array([r.dose for r in self.radio_plan.tx_visits])
    
    
    def truncate_radio_visits(self) -> None:
        """Pop the visits during radiotherapy from the patient data.
        """
        keep_idx = (self.visit_days > self.radio_days[-1]).nonzero()[0][0]
        self.visits = self.visits[keep_idx:]
        self.radio_plan = None
    
    
    def _unpack_ct_visit(self, ct_visit) -> ChemotherapyTreatment:
        date_str = ct_visit["time"].split("T")[0]
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        date_obj = date_obj.date()
        
        sf = 0.82  # default value
        
        return ChemotherapyTreatment(date_obj, sf)
    
    
    def _unpack_chemotherapy(self, info: dict) -> ChemotherapySpecification:
        """Unpack the visits from the JSON file.
        """
        ct_visits = []
        for ct_visit in info["chemotherapy"]:
            ct_visits.append(self._unpack_ct_visit(ct_visit))
        
        return ChemotherapySpecification(ct_visits)
    
    
    def get_chemotherapy_protocol(self) -> dict:
        """Get the Chemotherapy protocol.

        Returns:
            Dictionary of the chemotherapy protocol.
        """
        first = self.get_timeline()[0]
        return {days_since_first(r.time, first): r.sf for r in self.chemo_plan.tx_visits}
    
    
    @property
    def chemo_days(self) -> List[float]:
        """Get the number of days since the first visit for each chemotherapy visit.
        """
        first = self.get_timeline()[0]
        return np.array([days_since_first(r.time, first) for r in self.chemo_plan.tx_visits])


    @property
    def chemo_effects(self) -> List[float]:
        """Get the doses for each radiotherapy visit.
        """
        return np.array([r.sf for r in self.chemo_plan.tx_visits])
