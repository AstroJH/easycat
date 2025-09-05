from .core import LightcurveReprocessor

import astropy.units as u
import pandas as pd
from ...util import grp_by_max_interval, find_outliers, databinner, dbscan
import numpy as np

class ZTFReprocessor(LightcurveReprocessor):
    @classmethod
    def can_process(cls, metadata):
        return metadata.get("telescope") == "ZTF"
    
    def reprocess(self, lcurve, **kwargs):
        pos_ref = kwargs.get("pos_ref", None)
        dbscan_radius = kwargs.get("dbscan_radius", 0.5*u.arcsec)
        min_neighbors = kwargs.get("min_neighbors", 5)
        min_cluster_size = kwargs.get("min_cluster_size", 1)

        outlier_threshold = kwargs.get("outlier_threshold", 5)
        max_interval = kwargs.get("max_interval", 1.2)

        lcurve = lcurve.sort_values(by="mjd")
        lcurve = lcurve[lcurve["catflags"]==0]

        lcurve.reset_index(drop=True, inplace=True)

        if pos_ref is not None and len(lcurve) > 0:
            lcurve = dbscan.filter_dbscan(
                lcurve,
                pos_ref=pos_ref,
                radius=dbscan_radius,
                min_neighbors=min_neighbors,
                min_cluster_size=min_cluster_size
            )
        
        return lcurve
    

    def group(self, lcurve:pd.DataFrame, epoch_size=1):
        mjd = lcurve["mjd"].to_numpy()
        mag = lcurve["mag"].to_numpy()
        magerr = lcurve["magerr"].to_numpy()

        size = len(mjd)

        df = pd.DataFrame(data={
            "mjd": [],
            "mag": [],
            "magerr": []
        })

        i = 0
        begin = 0
        while begin < size:
            end = np.searchsorted(mjd, mjd[begin]+epoch_size, side="right")
            _mag = np.mean(mag[begin:end])
            _magerr = np.sqrt(np.sum(magerr[begin:end]**2))/(end-begin)
            _mjd = np.median(mjd[begin:end])

            df.loc[i] = (_mjd, _mag, _magerr)

            i += 1
            begin = end
        
        return df


    
    def batch_reprocess(self, catalog:pd.DataFrame):
        ...