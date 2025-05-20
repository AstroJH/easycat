import numpy as np
import pandas as pd

from astropy import units as u

from .core import LightcurveReprocessor
from ...util import grp_by_max_interval, find_outliers, databinner, dbscan

class WiseReprocessor(LightcurveReprocessor):
    def __init__(self):
        super().__init__()
        self._missing_check_fields = ["mjd", "w1mag", "w2mag", "w1sigmag", "w2sigmag", "na", "nb"]

    @classmethod
    def can_process(cls, metadata):
        return metadata.get("telescope") == "WISE"
    

    def reprocess(self, lcurve:pd.DataFrame, **kwargs):

        pos_ref = kwargs.get("pos_ref", None)
        dbscan_radius = kwargs.get("dbscan_radius", 0.5*u.arcsec)
        min_neighbors = kwargs.get("min_neighbors", 5)
        min_cluster_size = kwargs.get("min_cluster_size", 1)

        outlier_threshold = kwargs.get("outlier_threshold", 5)
        max_interval = kwargs.get("max_interval", 1.2)

        lcurve = lcurve.sort_values(by="mjd")
        lcurve = self.filter_missing(lcurve)
        lcurve = self.criteria_basic(lcurve)

        if pos_ref is not None:
            lcurve = dbscan.filter_dbscan(
                lcurve,
                pos_ref=pos_ref,
                radius=dbscan_radius,
                min_neighbors=min_neighbors,
                min_cluster_size=min_cluster_size
            )

        lcurve = self.filter_outliers(
            lcurve,
            outlier_threshold=outlier_threshold,
            max_interval=max_interval
        )
        
        return lcurve

    def criteria_basic(self, lcurve):
        """
        Parameters
        ----------
        lcurve : pd.DataFrame
            WISE lightcurve dataframe containing mandatory columns:
            - na: int
            - nb: int
            - saa_sep: float
            - qi_fact: int
            - qual_frame: int
            - moon_masked: str
            - cc_flags: str
            - w1rchi2: float
            - w2rchi2: float
        
        Returns
        -------
        pd.DataFrame
            Subset of lightcurve data passing all quality criteria
        
        Selection Criteria
        ------------------
        Quality Flags
            (qual_frame > 0 OR qual_frame == -1) AND qi_fact == 1
            - qual_frame: -1=valid single frame, >0=valid multi-frame coadd
            - qi_fact=1 selects highest quality data segments
            
        Measurement Stability
            na == 0 AND nb <= 2
            - Ensures no anomalous W1 measurements (na=0)
            - Limits W2 anomalies to ≤2 detections (nb≤2)
            
        Spacecraft Position
            saa_sep > 0
            - Excludes data during South Atlantic Anomaly (SAA) passage
            
        Lunar Contamination
            moon_masked[:2] == "00"
            - Filters data with lunar illumination artifacts
            
        Atmospheric Effects
            cc_flags[:2] == "00"
            - Removes cloud-contaminated observations
            
        Photometric Quality
            w1rchi2 < 5 AND w2rchi2 < 5
            - Ensures reliable photometric solutions
        
        References
        ----------
        """

        na = lcurve["na"]
        nb = lcurve["nb"]
        saa_sep = lcurve["saa_sep"]
        qi_fact = lcurve["qi_fact"]
        qual_frame = lcurve["qual_frame"]

        w1rchi2 = lcurve["w1rchi2"]
        w2rchi2 = lcurve["w2rchi2"]

        cond1 = ((qual_frame > 0) | (qual_frame == -1)) & (qi_fact == 1)
        cond2 = (na == 0) & (nb <= 2)
        cond3 = (saa_sep > 0)
        cond4 = lcurve["moon_masked"].apply(lambda s: s[:2]) == "00"
        cond5 = lcurve["cc_flags"].apply(lambda s: s[:2]) == "00"
        cond6 = (w1rchi2 < 5) & (w2rchi2 < 5)

        return lcurve[cond1 & cond2 & cond3 & cond4 & cond5 & cond6]
    

    def filter_missing(self, lcurve, missing_value=-1):
        """ Filters missing values for specified fields (missing values are represented by -1 by default). """

        fields = self._missing_check_fields
        
        mask = np.full(len(lcurve), True)

        for f in fields:
            mask = mask & (lcurve[f] != missing_value)
        
        return lcurve[mask]
    

    def filter_uncertainty(self, lcurve, w1threshold, w2threshold):
        w1sigmag = lcurve["w1sigmag"]
        w2sigmag = lcurve["w2sigmag"]

        mask = (w1sigmag <= w1threshold) & (w2sigmag <= w2threshold)
        
        return lcurve[mask]


    def filter_outliers(self, lcurve:pd.DataFrame,
        outlier_threshold=5, max_interval=1.2):

        mjd = lcurve["mjd"].to_numpy()
        los, his = grp_by_max_interval(mjd, max_interval)

        needremove = np.empty(0, dtype=np.intp)

        for lo, hi in zip(los, his):
            epoch = lcurve.iloc[lo:hi+1]
            outliers1 = find_outliers(epoch["w1mag"], outlier_threshold)
            outliers2 = find_outliers(epoch["w2mag"], outlier_threshold)
            outliers = np.union1d(outliers1, outliers2) + lo

            if len(outliers) > 0:
                needremove = np.concatenate([needremove, outliers])
        
        keep_indices = [i for i in range(len(lcurve)) if i not in needremove]

        lcurve = lcurve.iloc[keep_indices]
        lcurve.reset_index(drop=True, inplace=True)
        return lcurve
    

    def clean_epoch(self, lcurve, max_interval=1.2):
        mjd = lcurve["mjd"].to_numpy()
        los, his = grp_by_max_interval(mjd, max_interval=max_interval)
        
        mask = np.full_like(mjd, True, dtype=np.bool)
        # n_epoch = 0

        for lo, hi in zip(los, his):
            if hi - lo + 1 < 5:
                mask[lo:hi+1] = False
            # else:
            #     n_epoch += 1
        
        return lcurve[mask]



    def generate_longterm_lcurve(self, lcurve, max_interval=1.2, method="mean"):
        mjd = lcurve["mjd"].to_numpy()
        w1mag = lcurve["w1mag"].to_numpy()
        w2mag = lcurve["w2mag"].to_numpy()
        w1err = lcurve["w1sigmag"].to_numpy()
        w2err = lcurve["w2sigmag"].to_numpy()

        los, his = grp_by_max_interval(mjd, max_interval)

        def bindata(param):
            lo, hi = param
            hi += 1
            bin_mjd = np.median(mjd[lo:hi])
            bin_w1mag, bin_w1err = databinner(data=w1mag[lo:hi], sigmas=w1err[lo:hi], method=method)
            bin_w2mag, bin_w2err = databinner(data=w2mag[lo:hi], sigmas=w2err[lo:hi], method=method)

            return bin_mjd, bin_w1mag, bin_w1err, bin_w2mag, bin_w2err

        bin_lis = list(map(bindata, zip(los, his)))
        longterm_lcurve = pd.DataFrame(data=bin_lis, columns=[
            "mjd", "w1mag", "w1sigmag", "w2mag", "w2sigmag"
        ], dtype=np.float64)

        return longterm_lcurve
