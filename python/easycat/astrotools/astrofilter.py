from typing import Callable, Literal
from astropy import units as u
from astropy.units import Quantity
from astropy.io import votable
import numpy as np
import requests

import pickle
from os.path import join as pjoin
import os
from scipy.interpolate import interp1d

from io import StringIO
import pandas as pd

from .core import flux2mag

URL_SVO2 = "http://svo2.cab.inta-csic.es"
URL_SVO2_FPS = pjoin(URL_SVO2, "theory", "fps", "fps.php")
URL_SVO2_VEGA = pjoin(URL_SVO2, "theory", "fps", "morefiles", "vega.dat")
URL_SVO2_SUN = pjoin(URL_SVO2, "theory", "fps", "morefiles", "sun.dat")

def calc_pivot_wavelength():
    ...

class AstroFilter:
    def __init__(
        self,
        facility,
        instrument,
        band,
        filterID,
        wavelength,
        resp,
        magsys: Literal["AB", "Vega", "ST"],
        flux_zero: float,
        wavelength_eff: float
    ):
        self.facility = facility
        self.instrument = instrument
        self.band = band

        if filterID is None:
            self.filterID = f"{facility}_{instrument}.{band}"
        else:
            self.filterID = filterID

        self.resp = np.array([wavelength, resp])

        self.magsys = magsys
        self.flux_zero = flux_zero
        self.wavelength_eff = wavelength_eff

    def get_transmission_data(self):
        return self.resp
    
    def get_transmission(self, kind="linear"):
        wlen = self.resp[0]
        resp = self.resp[1]

        return interp1d(wlen, resp, kind=kind)
    
    @staticmethod
    def get_vega_spectrum():
        return AstroFilter.get_spectrum_from_svo(URL_SVO2_VEGA)
    
    @staticmethod
    def get_sun_spectrum():
        return AstroFilter.get_spectrum_from_svo(URL_SVO2_SUN)
    
    @staticmethod
    def get_spectrum_from_svo(url):
        resp = requests.get(url)
        if resp.status_code != 200: raise Exception()

        spec = pd.read_csv(
            StringIO(resp.content.decode('utf-8')),
            delimiter="\\s+",
            header=0,
            names=["wavelength", "flux"]
        )

        wlen = spec["wavelength"].to_numpy() * u.AA
        flux = spec["flux"].to_numpy() * u.erg / u.cm / u.cm / u.s / u.AA

        return wlen, flux


    @staticmethod
    def download_from_svo2(svo2_filterID: str):
        table = votable.parse(f"{URL_SVO2}?ID={svo2_filterID}")
        wlen_eff = table.get_field_by_id("WavelengthEff").value * u.AA
        wlen_eff = wlen_eff.to_value(u.cm)

        magsys = table.get_field_by_id("MagSys").value
        zeropoint = table.get_field_by_id("ZeroPoint").value

        table = table.get_first_table()
        table = table.to_table().to_pandas()

        wavelength = (table["Wavelength"].to_numpy() * u.AA).to_value(u.cm)
        resp = table["Transmission"].to_numpy()

        names = svo2_filterID.split("/")
        facility = names[0]
        filter_names = names[1].split(".")
        instrument = filter_names[0]
        band = filter_names[1]

        astro_filter = AstroFilter(
            facility=facility,
            instrument=instrument,
            band=band,
            filterID=None, # filterID = {facility}_{instrument}.{band}
            wavelength=wavelength,
            resp=resp,
            magsys=magsys,
            flux_zero=zeropoint,
            wavelength_eff=wlen_eff
        )

        return astro_filter
    
    # def model2mag(self, model: Callable) -> float:
    #     """
    #     Calculate the model magnitude with specified filter.
    #     """

    #     wlen = self.resp[0] * u.cm

    #     dw = wlen[1:] - wlen[:-1]
    #     wlen = wlen[:-1]
    #     R = self.resp[1, :-1]

    #     R_sum = np.sum(R * dw)

    #     S_lam = model(wlen) # erg s-1 cm-2 cm-1
    #     F_lam = np.sum(S_lam * R * dw) / R_sum
    #     F_nu = F_lam / const.c * ()

    #     f0 = self.flux_zero * u.Jy
    #     mag = flux2mag(flux, f0)

    #     return mag

class AstroFilterDB:
    def __init__(self, filter_db_path):
        if not os.path.exists(filter_db_path):
            os.mkdir(filter_db_path)

        elif not os.path.isdir(filter_db_path):
            raise Exception(f"{filter_db_path} is not a directory!")

        self._memory_db = {}
        self._store_path = filter_db_path

    def get_filter(self, filterID: str) -> AstroFilter:
        # load filter from memory
        memory = self._memory_db
        astro_filter = memory.get(filterID, None)
        if astro_filter is not None:
            return astro_filter
        
        # load filter from local storage
        store_path = self._store_path
        filter_path = pjoin(store_path, filterID + ".pkl")
        if os.path.exists(filter_path):
            with open(filter_path, "rb") as file:
                astro_filter = pickle.load(file)

            memory.update({ filterID: astro_filter })
            return astro_filter

        # download from SVO2
        try:
            svo2_filterID = "/".join(filterID.split("_"))
            print(f"{filterID}: Downloading from SVO2... (SVO2 ID: {svo2_filterID})")

            astro_filter = AstroFilter.download_from_svo2(svo2_filterID)

            memory.update({ filterID: astro_filter })
            return astro_filter
        except:
            print(f"Failed. {filterID} is not found :(")

        return None


    def register(self, filterID: str, astro_filter: AstroFilter, override: bool = False):
        memory = self._memory_db
        if (memory.get(filterID) is not None) and (not override):
            raise Exception(f"{filterID} already exists.")
        
        memory.update({ filterID: astro_filter })


    def persist(self, override: bool = False):
        store_path = self._store_path
        memory = self._memory_db
        for filterID, astro_filter in memory.items():
            filepath = pjoin(store_path, filterID + ".pkl")

            if os.path.exists(filepath) and (not override):
                continue

            with open(filepath, "wb") as file:
                pickle.dump(astro_filter, file)
