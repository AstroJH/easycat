from os import path
from typing import Optional
import logging

from astroquery.sdss import SDSS
from astropy.units import Quantity
import astropy.units as u
from astropy.coordinates import SkyCoord
from . import core
from pandas import DataFrame

class SdssSpectrumDownloader:
    def __init__(self, radius:Quantity, store_dir:str,
                 catalog:Optional[DataFrame]=None, logpath:Optional[str]=None, n_works:int=10):
        self.catalog = catalog
        self.radius = radius
        self.store_dir = store_dir
        self.n_works = n_works
        self.logpath = logpath

    def query_from_network(self, coordinates:SkyCoord) -> DataFrame:
        coord = coordinates.transform_to(frame="fk5")
        xid = SDSS.query_region(coord, radius=self.radius, spectro=True)
        return xid
    
    def download_item(self, obj_id:str, param:dict) -> bool:
        raj2000 = param["raj2000"]
        dej2000 = param["dej2000"]
        coord = SkyCoord(raj2000, dej2000, unit="deg", frame="fk5")

        try:
            xid = self.query_from_network(coord)
        except Exception as e:
            logging.error(f"{obj_id}: An exception occurred while querying.", exc_info=True, stack_info=True)
            return False

        if len(xid) == 0: # no data
            logging.warning(f"{obj_id}: Empty data.")
            return True
        
        if len(xid) > 1:
            logging.error(f"{obj_id}: More sources.", exc_info=True, stack_info=True)
            return False
        
        try:
            sp = SDSS.get_spectra(matches=xid)[0]
        except Exception as e:
            logging.error(f"{obj_id}: An exception occurred while querying.", exc_info=True, stack_info=True)
            return False
        

        try:
            filepath = path.join(self.store_dir, obj_id+".fits")
            sp.writeto(filepath, overwrite=True)
        except Exception as e:
            logging.error(f"{obj_id}: store exception", exc_info=True, stack_info=True)
            return False
        
        return True
    
    def download(self, obj_id:Optional[str]=None, raj2000:Optional[float]=None, dej2000:Optional[float]=None):
        if obj_id is None:
            core.start(logpath=self.logpath, catalog=self.catalog,
                        handler=self.download_item,
                        n_workers=self.n_works)
        else:
            return self.download_item(obj_id=obj_id, param={
                "raj2000": raj2000,
                "dej2000": dej2000
            })
