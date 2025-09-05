from os import path
import logging

from astroquery.sdss import SDSS
from astropy.units import Quantity
import astropy.units as u
from astropy.coordinates import SkyCoord
from . import core
from pandas import DataFrame

from os.path import join as pjoin

from io import StringIO
from typing import Optional, Tuple, Literal
from warnings import deprecated

import requests
from astropy.table import Table
from . import core
from ..parallel import TaskDispatcher
import pandas as pd


BASEURL = "https://skyserver.sdss.org/"


def get_specurl(*, action:Literal["search", "pull"],
                specid:str, release:str="dr18", 
                radius:Quantity, ra, dec):
    
    url = pjoin(BASEURL, release)

    if action == "search":
        R = radius.to_value("arcmin")
        url = pjoin(url, "SkyServerWS", "SpectroQuery", "ConeSpectro")
        url = f"{url}?radius={R}&ra={ra}&dec={dec}&format=csv"
        return url
    
    if action == "pull":
        url = pjoin(url, "en", "get", "specById.ashx")
        url = f"{url}?ID={specid}"
        return url
    
    raise Exception(f"Action `{action}` is unknown.")


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


class SDSSDataArchive:
    def retrieve_photo_for_item(self,
        item_id:str,
        coord:SkyCoord,
        *,
        radius:Quantity=3*u.arcsec
    ) -> Tuple[bool, Optional[DataFrame]]:
        
        base_url = "http://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"

        def cmd(ra, dec, radius):
            sql = f"SELECT p.ObjID,p.ra,p.dec,p.u,p.g,p.r,p.i,p.z FROM photoObj p JOIN dbo.fGetNearbyObjEq({ra},{dec},{radius}) n ON n.objID=p.objID"
            sql = sql.replace(" ", "%20")
            return sql


        try:
            coord = coord.transform_to(frame="fk5")
            ra = coord.ra.degree
            dec = coord.dec.degree
            radius = radius.to_value("arcmin")

            url = f"{base_url}?cmd={cmd(ra, dec, radius)}&format=csv"
            content = "\n".join(requests.get(url).content.decode('utf-8').splitlines()[1:])
            data = pd.read_csv(StringIO(content))
        except:
            logging.error(f"{item_id}: An exception occurred while querying.")
            return False, None


        return True, data
        

    def retrieve_photo(self, *,
        catalog:DataFrame,
        radius:Quantity=3*u.arcsec,
        checkpoint:Optional[str]=None,
        n_works:int=4
    ):
        
        def task(item_id, param):
            raj2000 = param.get("raj2000")
            dej2000 = param.get("dej2000")
            coord = SkyCoord(raj2000, dej2000, unit="deg", frame="fk5")

            return self.retrieve_photo_for_item(
                item_id, coord,
                radius=radius
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
    