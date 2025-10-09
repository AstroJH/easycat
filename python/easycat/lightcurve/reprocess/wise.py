import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Literal

from astropy import units as u
from astropy.units import Quantity
from astropy import constants as const
from scipy.optimize import fsolve
from astropy.cosmology import FlatLambdaCDM

from .core import LightcurveReprocessor
from ...util import grp_by_max_interval, find_outliers, databinner, dbscan

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
lambda_w1 = 3.4*u.um
lambda_w2 = 4.6*u.um
nu_w1 = (const.c/lambda_w1).to(u.GHz)
nu_w2 = (const.c/lambda_w2).to(u.GHz)


def get_flux_zero(band:Literal["W1", "W2"]) -> Quantity:
    if band == "W1":
        # f0 = 306.681 * u.Jy
        f0 = 309.540 * u.Jy
    elif band == "W2":
        # f0 = 170.663 * u.Jy
        f0 = 171.787 * u.Jy
    else:
        raise Exception(f"Error band: {band}")
    
    return f0


def mag2flux(mag:float, band:Literal["W1", "W2"]) -> Quantity:
    f0 = get_flux_zero(band)
    flux = f0 / (10**(0.4*mag))
    return flux.to(u.Jy)


def flux2mag(flux:Quantity, band:Literal["W1", "W2"]) -> float:
    f0 = get_flux_zero(band)
    mag = 2.5 * np.log10(f0/flux)
    return mag.to_value()


def planck(nu:Quantity, T:Quantity) -> Quantity:
    h = const.h
    c = const.c
    k_B = const.k_B

    A = 2*h*nu**3/c**2

    B = np.exp(h*nu/k_B/T) - 1

    return (A/B).to(u.Jy)


def flux_obs(nu_obs:Quantity, scale:float, T:Quantity, z:float) -> Quantity:
    nu_emit = nu_obs * (1+z)
    dL = cosmo.luminosity_distance(z)

    flux = (1+z) * (scale*u.cm*u.cm) * planck(nu_emit, T) / (4*np.pi*dL*dL)
    return flux.to(u.Jy)


def get_bb_equation(w1flux:Quantity, w2flux:Quantity, z:float):
    w1flux_val = w1flux.to_value(u.Jy)
    w2flux_val = w2flux.to_value(u.Jy)

    def equation(T:float):
        T_unit = T * u.K
        return w1flux_val*planck((1+z)*nu_w2, T_unit).to_value(u.Jy) - \
            w2flux_val*planck((1+z)*nu_w1, T_unit).to_value(u.Jy)
    return equation


def solve_bbody(w1flux:Quantity, w2flux:Quantity, z:float, T0:Quantity=1.5e3*u.K):
    # w1flux:float = w1flux.to_value(u.Jy)
    # w2flux:float = w2flux.to_value(u.Jy)

    # def equation(T):
    #     T_unit = T * u.K
    #     return w1flux*planck((1+z)*nu_w2, T_unit) - \
    #         w2flux*planck((1+z)*nu_w1, T_unit)
    
    equation = get_bb_equation(w1flux, w2flux, z)
    
    temperature = fsolve(equation, T0.to_value(u.K), maxfev=1000)[0]
    dL = cosmo.luminosity_distance(z)
    scale = w1flux * 4*np.pi*dL*dL / (1+z) / planck((1+z)*nu_w1, temperature*u.K)

    return scale.to_value(u.cm*u.cm), temperature


class WISEReprocessor(LightcurveReprocessor):
    def __init__(self):
        super().__init__()
        self._missing_check_fields = ["mjd", "w1mag", "w2mag", "w1sigmag", "w2sigmag", "na", "nb"]

    @classmethod
    def can_process(cls, metadata):
        return metadata.get("telescope") == "WISE"
    

    def reprocess(self, lcurve:DataFrame, **kwargs):

        pos_ref = kwargs.get("pos_ref", None)
        dbscan_radius = kwargs.get("dbscan_radius", 0.5*u.arcsec)
        min_neighbors = kwargs.get("min_neighbors", 5)
        min_cluster_size = kwargs.get("min_cluster_size", 1)

        outlier_threshold = kwargs.get("outlier_threshold", 5)
        max_interval = kwargs.get("max_interval", 1.2)

        lcurve = lcurve.sort_values(by="mjd")
        lcurve.reset_index(drop=True, inplace=True)

        lcurve = self.filter_missing(lcurve)
        lcurve = self.criteria_basic(lcurve)

        if pos_ref is not None and len(lcurve) > 0:
            lcurve = dbscan.filter_dbscan(
                lcurve,
                pos_ref=pos_ref,
                radius=dbscan_radius,
                min_neighbors=min_neighbors,
                min_cluster_size=min_cluster_size
            )

        if len(lcurve) > 0:
            lcurve = self.filter_outliers(
                lcurve,
                outlier_threshold=outlier_threshold,
                max_interval=max_interval
            )
        
        return lcurve

    def criteria_basic(self, lcurve:DataFrame):
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

        lcurve = lcurve[cond1 & cond2 & cond3 & cond4 & cond5 & cond6]
        lcurve.reset_index(drop=True, inplace=True)
        return lcurve


    def filter_missing(self, lcurve:DataFrame, missing_value=-1):
        """ Filters missing values for specified fields (missing values are represented by -1 by default). """

        fields = self._missing_check_fields
        
        mask = np.full(len(lcurve), True)

        for f in fields:
            mask = mask & (lcurve[f] != missing_value)
        
        lcurve = lcurve[mask]
        lcurve.reset_index(drop=True, inplace=True)

        return lcurve
    

    def filter_uncertainty(self, lcurve:DataFrame, w1threshold, w2threshold):
        w1sigmag = lcurve["w1sigmag"]
        w2sigmag = lcurve["w2sigmag"]

        mask = (w1sigmag <= w1threshold) & (w2sigmag <= w2threshold)
        
        return lcurve[mask]


    def filter_outliers(self, lcurve:DataFrame,
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
    

    def clean_epoch(self, lcurve:DataFrame, n_least=5, max_interval=1.2):
        mjd = lcurve["mjd"].to_numpy()
        los, his = grp_by_max_interval(mjd, max_interval=max_interval)
        
        mask = np.full_like(mjd, True, dtype=np.bool)
        # n_epoch = 0

        for lo, hi in zip(los, his):
            if hi - lo + 1 < n_least:
                mask[lo:hi+1] = False
            # else:
            #     n_epoch += 1
        
        lcurve = lcurve[mask]
        lcurve.reset_index(drop=True, inplace=True)
        return lcurve


    def generate_longterm_lcurve(self, lcurve:DataFrame, max_interval=1.2, method="mean"):
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

            subw1mag = w1mag[lo:hi]
            subw2mag = w2mag[lo:hi]
            subw1err = w1err[lo:hi]
            subw2err = w2err[lo:hi]

            subw1flux = mag2flux(subw1mag, "W1").value
            subw2flux = mag2flux(subw2mag, "W2").value

            bin_w1flux, _ = databinner(subw1flux, None, method=method)
            bin_w2flux, _ = databinner(subw2flux, None, method=method)
            
            bin_w1mag = flux2mag(bin_w1flux*u.Jy, "W1")
            bin_w2mag = flux2mag(bin_w2flux*u.Jy, "W2")

            N = len(subw1mag)
            bin_w1err = np.sqrt(
                np.var(subw1mag) +
                np.sum(subw1err*subw1err)/N/N +
                0.016*0.016/N
            )/np.sqrt(N)

            N = len(subw2mag)
            bin_w2err = np.sqrt(
                np.var(subw2mag) +
                np.sum(subw2err*subw2err)/N/N +
                0.016*0.016/N
            )/np.sqrt(N)
            
            return bin_mjd, bin_w1mag, bin_w1err, bin_w2mag, bin_w2err

        bin_lis = list(map(bindata, zip(los, his)))
        longterm_lcurve = pd.DataFrame(data=bin_lis, columns=[
            "mjd", "w1mag", "w1sigmag", "w2mag", "w2sigmag"
        ], dtype=np.float64)

        return longterm_lcurve
    
    def get_quiescent_part(self, lcurve:DataFrame, *,
        initdiff=0.2, diff_mu=0.1, diff_s=1.635):
        w1qstate = self._get_quiescent_mask(lcurve, "w1mag", initdiff=initdiff, diff_mu=diff_mu, diff_s=diff_s)
        w2qstate = self._get_quiescent_mask(lcurve, "w2mag", initdiff=initdiff, diff_mu=diff_mu, diff_s=diff_s)

        qstate = w1qstate & w2qstate

        return lcurve[qstate]

    def _get_quiescent_mask(self, lcurve:DataFrame, valname:str, *,
        initdiff=0.2, diff_mu=0.1, diff_s=1.635):

        vals = lcurve[valname]
        length = len(vals)

        qstate = np.full(length, fill_value=False, dtype=np.bool)

        # initial quiescent state
        qval = np.max(vals)
        mask = np.abs(vals - qval) < initdiff
        qstate[mask] = True

        # iterative algorithm
        flag = (np.sum(qstate) != length)

        while flag:
            qvals = vals[qstate]
            mu = np.mean(qvals)
            s = np.std(qvals)

            diff = np.abs(vals - mu)
            mask = (~qstate) & (diff<diff_mu) & (diff<diff_s*s)

            qstate[mask] = True

            # New points append into quiescent state
            # , and qstate isn't full
            # => let's continue!
            flag = (np.sum(mask)!=0) & (np.sum(qstate)!=length)
        
        return qstate
    

    def get_bbody_params(self, lcurve:DataFrame, redshift:float=0.):
        results = []

        for _, row in lcurve.iterrows():
            w1mag = row["w1mag"]
            w2mag = row["w2mag"]

            w1flux = mag2flux(w1mag, "W1")
            w2flux = mag2flux(w2mag, "W2")

            res = solve_bbody(w1flux, w2flux, z=redshift)
            results.append(res)
        return results

    def kcorrect(self, lcurve:DataFrame, redshift:float=0.):
        dL = cosmo.luminosity_distance(redshift)
        area = 4*np.pi*dL*dL

        w1qstate = self._get_quiescent_mask(lcurve, "w1mag")
        w2qstate = self._get_quiescent_mask(lcurve, "w2mag")

        qstate = w1qstate & w2qstate
        qlcurve = lcurve[qstate] # quiescent state
        # hlcurve = lcurve[~qstate] # high state

        qw1mag = np.mean(qlcurve["w1mag"])
        qw2mag = np.mean(qlcurve["w2mag"])

        qw1flux = mag2flux(qw1mag, "W1")
        qw2flux = mag2flux(qw2mag, "W2")

        qscale, qtemp = solve_bbody(qw1flux, qw2flux, z=redshift)
        qA = qscale*u.cm*u.cm

        w1mag_kcorr = np.empty(len(lcurve), dtype=np.float64)
        w2mag_kcorr = np.empty(len(lcurve), dtype=np.float64)

        for isq, (i, row) in zip(qstate, lcurve.iterrows()):
            w1flux = mag2flux(row["w1mag"], "W1")
            w2flux = mag2flux(row["w2mag"], "W2")

            # try to calculate BBody equation
            scale, T = solve_bbody(w1flux, w2flux, redshift)
            equation = get_bb_equation(w1flux, w2flux, redshift)

            if isq or (np.abs(equation(T)) < 0.01):
                A = scale*u.cm*u.cm
                w1flux_rest = A * planck(nu_w1, T*u.K) / area
                w2flux_rest = A * planck(nu_w2, T*u.K) / area
            else:
                scale, T = solve_bbody(w1flux-qw1flux, w2flux-qw2flux, redshift)
                A = scale*u.cm*u.cm
                
                equation = get_bb_equation(w1flux-qw1flux, w2flux-qw2flux, redshift)
                if np.abs(equation(T)) < 0.01:
                    w1flux_rest = (A * planck(nu_w1, T*u.K) + qA * planck(nu_w1, qtemp*u.K)) / area
                    w2flux_rest = (A * planck(nu_w2, T*u.K) + qA * planck(nu_w2, qtemp*u.K)) / area
                else:
                    w1flux_rest = np.nan*u.Jy
                    w2flux_rest = np.nan*u.Jy
            w1mag_kcorr[i] = flux2mag(w1flux_rest, "W1")
            w2mag_kcorr[i] = flux2mag(w2flux_rest, "W2")

        return w1mag_kcorr, w2mag_kcorr