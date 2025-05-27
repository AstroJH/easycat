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
from pandas import DataFrame
from ..parallel import TaskDispatcher

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


def combine_wisedata(t_neowise:Table, t_allwise:Table) -> Table:
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


class WiseDataDownloader:
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

    def download_item(self, obj_id:str, param:dict, *,
                      return_data:bool=False,
                      store_data:bool=True):
        return_content = None
        raj2000 = param["raj2000"]
        dej2000 = param["dej2000"]
        coord = SkyCoord(raj2000, dej2000, unit="deg", frame="fk5")

        # download WISE data
        try:
            t_allwise  = self.query_from_network(coord, "allwise")
            t_neowise  = self.query_from_network(coord, "neowise")
            t_combined = combine_wisedata(t_neowise=t_neowise, t_allwise=t_allwise)
            if return_data: return_content = t_combined.to_pandas()
        except Exception as e:
            logging.error(f"{obj_id}: An exception occurred while querying.", exc_info=True, stack_info=True)
            return False, return_content

        if len(t_combined) == 0: # no data
            logging.warning(f"{obj_id}: Empty data.")
            if return_data: return_content = t_combined.to_pandas()
            return True, return_content
        
        # store WISE data to local disk
        if store_data:
            try:
                filepath = path.join(self.store_dir, obj_id+".fits")
                t_combined.write(filepath, overwrite=True)
            except Exception as e:
                logging.error(f"{obj_id}: store exception", exc_info=True, stack_info=True)
                return False, return_content
        
        return True, return_content
    
    def download(self,
        pos_ref:Optional[SkyCoord]=None,
        mode:Literal["item", "catalog"]="catalog",
        obj_id:Optional[str]=None,
        return_data:bool=False,
        store_data:bool=True
    ):

        def task(obj_id, param):
            return self.download_item(obj_id, param, return_data=return_data, store_data=store_data)

        if mode == "catalog":
            record = TaskDispatcher(
                catalog=self.catalog,
                task=task,
                checkpoint=self.logpath,
                n_workers=self.n_works,
                mode="thread",
                rehandle_failed=True
            ).dispatch()
            return record
        
        raj2000 = pos_ref.ra.to("deg").value
        dej2000 = pos_ref.dec.to("deg").value

        return task(obj_id=obj_id, param={
            "raj2000": raj2000,
            "dej2000": dej2000
        })

class WISEDataArchive:
    def __init__(self):
        self.combined_fields = [
            "raj2000", "dej2000", "mjd",
            "w1mag", "w1sigmag", "w1rchi2",
            "w2mag", "w2sigmag", "w2rchi2",
            "na", "nb", "qi_fact", "cc_flags",
            "qual_frame", "saa_sep", "moon_masked"
        ]
    
    def download_wisetable(self, coords:SkyCoord, table:Literal["allwise", "neowise"], radius:Quantity) -> Table:
        info = WISECAT_INFO.get(table, None)
        if info is None: raise InvalidQueryError("Catalog name must be allwise or neowise!")

        catname = info.get("catname")
        columns = ",".join(info.get("colnames"))

        result:Table = Irsa.query_region(coords, catalog=catname, columns=columns,
                            spatial="Cone", radius=radius)

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


    def retrieve_photo_for_item(self,
        item_id:str,
        coord:SkyCoord,
        *,
        radius:Quantity=6*u.arcsec,
        store_dir:Optional[str]=None,
        return_data:bool=True
    ) -> tuple[bool, Optional[DataFrame]]:

        # download WISE data
        try:
            t_allwise  = self.download_wisetable(coord, "allwise", radius)
            t_neowise  = self.download_wisetable(coord, "neowise", radius)
            t_combined = combine_wisedata(t_neowise=t_neowise, t_allwise=t_allwise)
        except:
            logging.error(f"{item_id}: An exception occurred while querying.")
            return False, None


        if len(t_combined) == 0: # no data
            logging.warning(f"{item_id}: Empty data.")
        
        # store WISE data to local disk
        if store_dir is not None:
            try:
                filepath = path.join(store_dir, item_id+".fits")
                t_combined.write(filepath, overwrite=True)
            except:
                logging.error(f"{item_id}: store exception")
                return False, None
        
        
        return True, \
            None if not return_data else t_combined.to_pandas()
        

    def retrieve_photo(self, *,
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

            return self.retrieve_photo_for_item(
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