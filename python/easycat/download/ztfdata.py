from os import path
from io import StringIO
from typing import Optional, Tuple
import logging

# from ztfquery import lightcurve
import requests
from astropy.units import Quantity
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from . import core
from pandas import DataFrame
from ..parallel import TaskDispatcher
import pandas as pd

BASEURL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"


class ZTFLightcurveDownloader:
    def __init__(self, radius:Quantity, store_dir:str,
                 catalog:Optional[DataFrame]=None, logpath:Optional[str]=None, n_works:int=10):
        self.catalog = catalog
        self.radius = radius
        self.store_dir = store_dir
        self.n_works = n_works
        self.logpath = logpath

    def query_from_network(self, coordinates:SkyCoord) -> DataFrame:
        coord = coordinates.transform_to(frame="fk5")
        raj2000 = coord.ra.degree
        dej2000 = coord.dec.degree
        radius = self.radius.to_value("deg")

        url = f"{BASEURL}?POS=CIRCLE {raj2000} {dej2000} {radius}&FORMAT=CSV"
        url = url.replace(" ", "%20")

        # lcq = lightcurve.LCQuery.from_position(raj2000, dej2000, self.radius.to_value(u.arcsecond))
        # return lcq.data
        return pd.read_csv(
            StringIO(
                requests.get(url).content.decode('utf-8')
            )
        )
    
    def download_item(self, obj_id:str, param:dict) -> bool:
        raj2000 = param["raj2000"]
        dej2000 = param["dej2000"]
        coord = SkyCoord(raj2000, dej2000, unit="deg", frame="fk5")

        # download ZTF lightcurve
        try:
            df_ztf = self.query_from_network(coord)
        except Exception as e:
            logging.error(f"{obj_id}: An exception occurred while querying.", exc_info=True, stack_info=True)
            return False

        if len(df_ztf) == 0: # no data
            logging.warning(f"{obj_id}: Empty data.")
            return True
        
        # store ZTF lightcurve to local disk
        try:
            filepath = path.join(self.store_dir, obj_id+".csv")
            df_ztf.to_csv(filepath, index=False)
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


class ZTFDataArchive:
    def __init__(self): ...


    def retrieve_lcurve_for_item(self,
        item_id:str,
        coord:SkyCoord,
        *,
        radius:Quantity=6*u.arcsec,
        store_dir:Optional[str]=None,
        return_data:bool=True
    ) -> Tuple[bool, Optional[DataFrame]]:
        coord = coord.transform_to(frame="fk5")
        raj2000 = coord.ra.degree
        dej2000 = coord.dec.degree
        radius = radius.to_value("deg")

        url = f"{BASEURL}?POS=CIRCLE {raj2000} {dej2000} {radius}&FORMAT=CSV"
        url = url.replace(" ", "%20")

        try:
            lcurve = pd.read_csv(
                StringIO(
                    requests.get(url).content.decode('utf-8')
                )
            )
        except:
            logging.error(f"{item_id}: An exception occurred while querying.")
            return False, None

        # store ZTF lightcurve to local disk
        if store_dir is not None:
            try:
                filepath = path.join(store_dir, item_id+".fits")
                Table.from_pandas(lcurve, index=False).write(filepath, overwrite=True)
            except:
                logging.error(f"{item_id}: store exception")
                return False, None

        return True, \
            None if not return_data else lcurve
        

    def retrieve_lcurve(self, *,
        catalog:DataFrame,
        radius:Quantity=6*u.arcsec,
        return_data:bool=True,
        store_dir:Optional[str]=None,
        checkpoint:Optional[str]=None,
        n_works:int=4
    ):
        
        def task(item_id, param):
            raj2000 = param.get("raj2000")
            dej2000 = param.get("dej2000")
            coord = SkyCoord(raj2000, dej2000, unit="deg", frame="fk5")

            return self.retrieve_lcurve_for_item(
                item_id, coord,
                radius=radius,
                store_dir=store_dir,
                return_data=return_data
            )

        record = TaskDispatcher(
            catalog=catalog,
            task=task,
            checkpoint=checkpoint,
            n_workers=n_works,
            mode="thread",
            rehandle_failed=True
        ).dispatch()
        return record