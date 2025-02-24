from os import path
from typing import Literal, Optional
import logging

from astroquery.ipac.irsa import Irsa
from astroquery.exceptions import InvalidQueryError
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.table import Table
import astropy.units as u
import easycat
from pandas import DataFrame

WISECAT_INFO = {
    "allwise": {
        "catname": "allwise_p3as_mep",
        "colnames": [
            "ra", "dec", "mjd", "w1mpro_ep", "w1sigmpro_ep", "w1rchi2_ep", "w2mpro_ep", "w2sigmpro_ep", "w2rchi2_ep",
            "na", "nb", "qi_fact", "cc_flags", "saa_sep", "moon_masked"
        ]
    },
    "neowise": {
        "catname": "neowiser_p1bs_psd",
        "colnames": [
            "ra", "dec", "mjd", "w1mpro", "w1sigmpro", "w1rchi2", "w2mpro", "w2sigmpro", "w2rchi2",
            "na", "nb", "qi_fact", "cc_flags", "qual_frame", "saa_sep", "moon_masked"
        ]
    }
}


def combine_wisedata(t_neowise:Table, t_allwise:Table):
    fields = [
        "raj2000", "dej2000", "mjd",
        "w1mag", "w1sigmag", "w1rchi2",
        "w2mag", "w2sigmag", "w2rchi2",
        "na", "nb", "qi_fact", "cc_flags",
        "qual_frame", "saa_sep", "moon_masked"
    ]

    t_allwise.rename_columns(["ra", "dec", "mjd",
                            "w1mpro_ep", "w1sigmpro_ep", "w1rchi2_ep",
                            "w2mpro_ep", "w2sigmpro_ep", "w2rchi2_ep",
                            "na", "nb", "qi_fact", "cc_flags",
                            "qual_frame", "saa_sep", "moon_masked"], fields)
    
    t_neowise.rename_columns(["ra", "dec", "mjd",
                            "w1mpro", "w1sigmpro", "w1rchi2",
                            "w2mpro", "w2sigmpro", "w2rchi2",
                            "na", "nb", "qi_fact", "cc_flags",
                            "qual_frame", "saa_sep", "moon_masked"], fields)

    t_allwise.keep_columns(fields)
    t_neowise.keep_columns(fields)

    t_combined:Table = vstack([t_allwise, t_neowise], metadata_conflicts="silent")

    # the normal values for the following fields should be >= 0,
    # so we use -1 to fill masked/abnormal value
    # 
    # > Why don't use `NaN` (i.e. Not-a-Number)?
    # [J.H. Wu] Just because I prefer numbers ᐛ
    t_combined["w1mag"]    = t_combined["w1mag"]   .filled(fill_value=-1)
    t_combined["w2mag"]    = t_combined["w2mag"]   .filled(fill_value=-1)
    t_combined["w1sigmag"] = t_combined["w1sigmag"].filled(fill_value=-1)
    t_combined["w2sigmag"] = t_combined["w2sigmag"].filled(fill_value=-1)
    t_combined["w1rchi2"]  = t_combined["w1rchi2"] .filled(fill_value=-1)
    t_combined["w2rchi2"]  = t_combined["w2rchi2"] .filled(fill_value=-1)

    # NOTE The result will not be sorted.
    # which is better? :-I
    return t_combined


class WISEDataDownloader:
    def __init__(self, radius:Quantity, store_dir:str,
                 catalog:Optional[DataFrame]=None, logpath:Optional[str]=None, n_works:int=10):
        self.catalog = catalog
        self.radius = radius
        self.store_dir = store_dir
        self.n_works = n_works
        self.logpath = logpath

    def query_from_network(self, coordinates:SkyCoord, table:Literal["allwise", "neowise"]):
        info = WISECAT_INFO.get(table, None)
        if info is None: raise InvalidQueryError("Catalog name must be allwise or neowise!")

        catname = info.get("catname")
        columns = ",".join(info.get("colnames"))

        result:Table = Irsa.query_region(coordinates, catalog=catname, columns=columns,
                                         spatial="Cone", radius=self.radius)

        if table == "allwise":
            if len(result) > 0: result["qual_frame"] = -1
            else:               result.add_column(col=[], name="qual_frame")

        # -------------------------------
        for name in result.colnames:
            if result.dtype[name] == object:
                result[name] = result[name].astype(str)

            if name == "mjd":
                result[name].unit = u.day

        return result

    def download_item(self, obj_id:str, param:dict) -> bool:
        raj2000 = param["raj2000"]
        dej2000 = param["dej2000"]
        coord = SkyCoord(raj2000, dej2000, unit="deg", frame="fk5")
        radius = self.radius

        # download WISE data
        try:
            t_allwise  = self.query_from_network(coord, radius, "allwise")
            t_neowise  = self.query_from_network(coord, radius, "neowise")
            t_combined = combine_wisedata(t_neowise=t_neowise, t_allwise=t_allwise)
        except Exception as e:
            logging.error(f"{obj_id}: An exception occurred while querying.", exc_info=True, stack_info=True)
            return False

        if len(t_combined) == 0: # no data
            logging.warning(f"{obj_id}: Empty data.")
            return True
        
        # store WISE data to local disk
        try:
            filepath = path.join(self.store_dir, obj_id+".fits")
            t_combined.write(filepath, overwrite=True)
        except Exception as e:
            logging.error(f"{obj_id}: store exception", exc_info=True, stack_info=True)
            return False
        
        return True
    
    def download(self, obj_id:Optional[str]=None, raj2000:Optional[float]=None, dej2000:Optional[float]=None):
        if obj_id is None:
            easycat.start(logpath=self.logpath, catalog=self.catalog,
                        handler=self.download_item,
                        n_workers=self.n_works)
        else:
            self.download_item(obj_id=obj_id, param={
                "raj2000": raj2000,
                "dej2000": dej2000
            })