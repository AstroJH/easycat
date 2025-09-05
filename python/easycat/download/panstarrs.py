from os import path
from io import StringIO
from typing import Optional, Tuple, Literal
import logging
from warnings import deprecated

import requests
from astropy.units import Quantity
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from . import core
from pandas import DataFrame
from ..parallel import TaskDispatcher
import pandas as pd
import numpy as np

VERSION_API = "v0.1"
BASEURL = f"https://catalogs.mast.stsci.edu/api/{VERSION_API}/panstarrs"

FILTER_ID_MAP = {
    "g": 1,
    "r": 2,
    "i": 3,
    "z": 4,
    "y": 5,
    "all": None
}

class PanStarrsArchive:
    def retrieve_photo_for_item(
        self,
        item_id:str,
        coord:SkyCoord,
        *,
        band:Literal["g","r","i","z","y","all"]="all",
        radius:Quantity=2*u.arcsec,
        release:Literal["dr1", "dr2"]="dr2",
        catalog_panstarrs:Literal["mean", "stack", "detection", "forced_mean"]="detection",
        store_dir:Optional[str]=None,
        return_data:bool=True
    ) -> Tuple[bool, Optional[DataFrame]]:
        filter_id = FILTER_ID_MAP.get(band, None)

        coord = coord.transform_to(frame="fk5")
        raj2000 = coord.ra.degree
        dej2000 = coord.dec.degree
        radius = radius.to_value("deg")

        url = f"{BASEURL}/{release}/{catalog_panstarrs}.csv?ra={raj2000}&dec={dej2000}&radius={radius}"
        if filter_id is not None:
            url += f"&filterID={filter_id}"

        url = url.replace(" ", "%20")

        try:
            resp = requests.get(url)
            if resp.status_code != 200: raise Exception()

            lcurve = pd.read_csv(
                StringIO(
                    resp.content.decode('utf-8')
                )
            )
        
            if catalog_panstarrs == "detection":
                lcurve.sort_values(by="obsTime", inplace=True)

                # convert flux in Jy to magnitudes
                mag = -2.5*np.log10(lcurve['psfFlux']) + 8.90

                lcurve["psfMag"] = mag


        except Exception as e:
            logging.error(f"{item_id}: An exception occurred while querying.")
            return False, None

        if store_dir is not None:
            try:
                filepath = path.join(store_dir, item_id+".fits")
                Table.from_pandas(lcurve, index=False).write(filepath, overwrite=True)
            except:
                logging.error(f"{item_id}: store exception")
                return False, None

        return True, \
            None if not return_data else lcurve
        

    def retrieve_photo(self, *,
        catalog:DataFrame,
        band:Literal["g","r","i","z","y","all"]="all",
        radius:Quantity=2*u.arcsec,
        release:Literal["dr1", "dr2"]="dr2",
        catalog_panstarrs:Literal["mean", "stack", "detection", "forced_mean"]="detection",
        return_data:bool=True,
        store_dir:Optional[str]=None,
        checkpoint:Optional[str]=None,
        n_works:int=4
    ):
        
        def task(item_id, param):
            raj2000 = param.get("raj2000")
            dej2000 = param.get("dej2000")
            coord = SkyCoord(raj2000, dej2000, unit="deg", frame="fk5")

            return self.retrieve_photo_for_item(
                item_id, coord,
                band=band,
                radius=radius,
                release=release,
                catalog_panstarrs=catalog_panstarrs,
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