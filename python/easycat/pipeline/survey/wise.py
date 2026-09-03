from pathlib import Path

from typing import Union, Tuple, Literal, Optional
from numpy.typing import ArrayLike, NDArray

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from astropy.io import fits
from astropy.table import Table

from easycat.astrofilter import FilterDB, mag2flux, flux2mag
from easycat.lightcurve.features import intrinsic_variability_amplitude
from easycat.pipeline import DataPacket, ProcessingNode

_FILTER_DB = FilterDB()
W1 = _FILTER_DB.get('WISE/WISE.W1')
W2 = _FILTER_DB.get('WISE/WISE.W2')


class WiseLcLoader(ProcessingNode):
    def __init__(self, name: str = 'loader@wise'):
        super().__init__(name, minsize=0)
    
    def process(self, data: DataPacket) -> DataPacket:
        filepath = data.metadata.get('filepath')

        with fits.open(filepath) as hdul:
            lcurve = Table(hdul[1].data).to_pandas()

        data.light_curve = lcurve
        return data


class WiseLcStorage(ProcessingNode):
    def __init__(
        self,
        name: str = 'storage@wise',
        fmt: Literal['csv', 'fits'] = 'fits'
    ):
        super().__init__(name)

        self.config['format'] = fmt
    
    def process(self, data: DataPacket) -> DataPacket:
        fmt = self.config.get('format')
        output: Union[str, Path] = data.metadata['storage_path']

        if output is None:
            data.add_error(
                "Missing 'storage_path' in packet metadata.",
                self.name,
            )
            return data

        lcurve = data.light_curve

        if lcurve is None or len(lcurve) <= 0:
            data.add_warning(
                'empty data', self.name
            )
            return data

        if fmt == 'csv':
            lcurve.to_csv(output, index=False)
        elif fmt == 'fits':
            table = Table.from_pandas(lcurve, index=False, units=None)
            table_name = 'WISELC'

            primary_hdu = fits.PrimaryHDU()

            table_hdu = fits.BinTableHDU(table, name=table_name)
            table_hdu.header['EXTNAME'] = table_name
            table_hdu.header['AUTHOR'] = 'Virjid'
            table_hdu.header['CREATED'] = pd.Timestamp.now().isoformat()

            hdul = fits.HDUList([primary_hdu, table_hdu])
            hdul.writeto(output, overwrite=True)
        else:
            data.add_error(f"Supported format: csv and fits", self.name)

        return data


class WisePreprocessNode(ProcessingNode):
    MISSING_CHECK_FIELDS = ["mjd", "w1mag", "w2mag", "w1sigmag", "w2sigmag", "na", "nb"]

    def __init__(
        self,
        name: str = 'preprocess@wise',
        missing_value: float = -1
    ):
        super().__init__(name)
        self.config['sort_column'] = 'mjd'
        self.config['missing_value'] = missing_value
    
    def process(self, data: DataPacket) -> DataPacket:
        lc = data.light_curve
        
        mask = pd.Series(True, index=lc.index)
        for field in WisePreprocessNode.MISSING_CHECK_FIELDS:
            if field in lc.columns:
                mask = mask & (lc[field] != self.config['missing_value'])
        
        lc = lc[mask].copy()
        lc.sort_values(by=self.config['sort_column'], inplace=True)
        lc.reset_index(drop=True, inplace=True)
        
        data.light_curve = lc

        # Record statistics
        data.metadata['sorted'] = True
        return data


class WiseBasicCriteriaNode(ProcessingNode):
    """Node for applying basic quality criteria to WISE data."""
    
    def __init__(
        self,
        name: str = "basic_criteria@wise", 
        max_nb: int = 2,
        max_w1rchi2: float = 5,
        max_w2rchi2: float = 5
    ):
        super().__init__(name)

        self.config.update({
            'max_nb': max_nb,
            'max_w1rchi2': max_w1rchi2,
            'max_w2rchi2': max_w2rchi2
        })
    
    def process(self, data: DataPacket) -> DataPacket:
        """Apply quality criteria to filter WISE data."""
        
        lc = data.light_curve
        
        max_nb = self.config['max_nb']
        max_w1rchi2 = self.config['max_w1rchi2']
        max_w2rchi2 = self.config['max_w2rchi2']
        
        # Define conditions based on WISE quality flags
        na = lc["na"]
        nb = lc["nb"]
        saa_sep = lc["saa_sep"]
        qi_fact = lc["qi_fact"]
        qual_frame = lc["qual_frame"]
        w1rchi2 = lc["w1rchi2"]
        w2rchi2 = lc["w2rchi2"]

        cond1 = ((qual_frame > 0) | (qual_frame == -1)) & (qi_fact == 1)
        cond2 = (na == 0) & (nb <= max_nb)
        cond3 = (saa_sep > 0)
        cond4 = lc['moon_masked'].apply(lambda s: s[:2]) == '00'
        cond5 = lc['cc_flags'].apply(lambda s: s[:2]) == '00'
        cond6 = (w1rchi2 <= max_w1rchi2) & (w2rchi2 <= max_w2rchi2)
        
        # Apply all conditions
        mask = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
        lc_filtered = lc[mask].copy()
        lc_filtered.reset_index(drop=True, inplace=True)
        
        # Record statistics
        removed = len(lc) - len(lc_filtered)
        data.add_result('removed', removed, self.name)

        data.light_curve = lc_filtered
        return data


class WiseLocalSlopeNode(ProcessingNode):

    """Node for calculating local WISE magnitude slopes."""

    def __init__(
        self,
        name: str = "local_slope@wise",
        x_years: float = 1.0,
        weighted: bool = True
    ):

        super().__init__(name)

        self.config.update({
            'x_years': x_years,
            'weighted': weighted
        })

    def process(self, data: DataPacket) -> DataPacket:
        """Calculate local W1/W2 magnitude slopes."""

        lc = data.light_curve

        x_years = self.config['x_years']
        weighted = self.config['weighted']

        # Convert half-window from years to days
        half_window = x_years * 365.25

        # Extract light-curve columns
        mjd = lc["mjd"].to_numpy(dtype=float)

        w1mag = lc["w1mag"].to_numpy(dtype=float)
        w2mag = lc["w2mag"].to_numpy(dtype=float)

        w1sigmag = lc["w1sigmag"].to_numpy(dtype=float)
        w2sigmag = lc["w2sigmag"].to_numpy(dtype=float)

        k_w1 = np.full(len(lc), np.nan)
        k_w2 = np.full(len(lc), np.nan)

        def fit_slope(time, mag, sigma):
            """Fit a local linear slope."""

            valid = (
                np.isfinite(time)
                & np.isfinite(mag)
            )

            if weighted:
                valid &= (
                    np.isfinite(sigma)
                    & (sigma > 0)
                )

            time = time[valid]
            mag = mag[valid]

            if weighted:
                sigma = sigma[valid]

            # At least two valid points are required
            if len(time) < 2:
                return np.nan

            # Avoid degenerate time coordinates
            if np.ptp(time) == 0:
                return np.nan

            try:
                if weighted:
                    return np.polyfit(
                        time,
                        mag,
                        1,
                        w=1.0 / sigma
                    )[0]

                return np.polyfit(
                    time,
                    mag,
                    1
                )[0]

            except (np.linalg.LinAlgError, ValueError):
                return np.nan

        # Calculate local slopes for each data point
        for i, t0 in enumerate(mjd):

            if not np.isfinite(t0):
                continue

            mask = (
                np.isfinite(mjd)
                & (mjd >= t0 - half_window)
                & (mjd <= t0 + half_window)
            )

            k_w1[i] = fit_slope(
                mjd[mask],
                w1mag[mask],
                w1sigmag[mask]
            )

            k_w2[i] = fit_slope(
                mjd[mask],
                w2mag[mask],
                w2sigmag[mask]
            )

        # Add local slopes to the light curve
        lc = lc.copy()

        lc["k_w1"] = k_w1
        lc["k_w2"] = k_w2

        data.light_curve = lc

        return data


class WiseAnalyzer(ProcessingNode):
    def __init__(self, name: str = 'analyzer', minsize: int = 1, N: int = 1000):
        ProcessingNode.__init__(self, name, minsize)
        self.N = N
        self.sigma_probs = {
            1: 0.6826895,  # 1σ: 68.27%
            2: 0.9544997,  # 2σ: 95.45%
            3: 0.9973002   # 3σ: 99.73%
        }
    
    def process(self, data: DataPacket) -> DataPacket:
        lc = data.light_curve
        mjd = lc.mjd.to_numpy()
        w2mag = lc.w2mag.to_numpy()
        w1mag = lc.w1mag.to_numpy()
        w1err = lc.w1sigmag.to_numpy()
        w2err = lc.w2sigmag.to_numpy()
        mirc = w1mag - w2mag

        data.add_result(key='size', value=len(lc), node_name=self.name)

        rho, pvalue = pearsonr(w1mag, mirc)
        mcv_k, _ = np.polyfit(w1mag, mirc, deg=1)
        data.add_result(key='mcv_w1', value=rho, node_name=self.name)
        data.add_result(key='mcv_w1pv', value=pvalue, node_name=self.name)
        data.add_result(key='mcv_w1k', value=mcv_k, node_name=self.name)

        rho, pvalue = pearsonr(w2mag, mirc)
        mcv_k, _ = np.polyfit(w2mag, mirc, deg=1)
        data.add_result(key='mcv_w2', value=rho, node_name=self.name)
        data.add_result(key='mcv_w2pv', value=pvalue, node_name=self.name)
        data.add_result(key='mcv_w2k', value=mcv_k, node_name=self.name)

        i_cmin = np.argmin(mirc)
        i_cmax = np.argmax(mirc)
        cmin = mirc[i_cmin]
        cmax = mirc[i_cmax]
        dt = np.abs(mjd[i_cmin]-mjd[i_cmax])
        data.add_result(key='cmin', value=cmin, node_name=self.name)
        data.add_result(key='cmax', value=cmax, node_name=self.name)
        data.add_result(key='dt4dc', value=dt, node_name=self.name)

        w1varamp = intrinsic_variability_amplitude(w1mag, w1err, False) # * np.sqrt(1+row.Z)
        w2varamp = intrinsic_variability_amplitude(w2mag, w2err, False) # * np.sqrt(1+row.Z)
        w1varerr = np.sqrt(np.mean(w1err**2))
        w2varerr = np.sqrt(np.mean(w2err**2))
        data.add_result(key='w1varamp', value=w1varamp, node_name=self.name)
        data.add_result(key='w2varamp', value=w2varamp, node_name=self.name)
        data.add_result(key='w1varerr', value=w1varerr, node_name=self.name)
        data.add_result(key='w2varerr', value=w2varerr, node_name=self.name)


        # N = self.N
        # rho = np.empty(N)
        # for i in range(N):
        #     new_w1mag = perturb(w1mag, w1err)
        #     new_w2mag = perturb(w2mag, w2err)

        #     new_color = new_w1mag - new_w2mag

        #     rho[i] = pearsonr(new_w1mag, new_color).statistic
        
        # rho = np.sort(rho)
        
        # intervals = {}
        # for sigma, prob in self.sigma_probs.items():
        #     tail_prob = (1 - prob) / 2
        #     lower = np.percentile(rho, tail_prob * 100)
        #     upper = np.percentile(rho, (1 - tail_prob) * 100)
        #     intervals[f"{sigma}sigma"] = (lower, upper)
        
        # data.add_result(key='median', value=np.median(rho), node_name=self.name)
        # data.add_result(key='mean', value=np.mean(rho), node_name=self.name)
        # data.add_result(key='std', value=np.std(rho, ddof=1), node_name=self.name)
        # data.add_result(key='intervals', value=intervals, node_name=self.name)

        return data


# def wisebinner(values, errors, **kwargs) -> Tuple[float, float]:
#     value_col = kwargs.get('value_col')

#     if 'w1' in value_col:
#         zp = W1.zp_vega
#     elif 'w2' in value_col:
#         zp = W2.zp_vega
#     else:
#         return np.nan, np.nan
    
#     flux = mag2flux(values, zp)
#     flux_avg = np.mean(flux)
#     avg = flux2mag(flux_avg, zp)

#     N = len(values)
#     var = np.sum((values - avg)**2) / (N - 1) + np.sum((errors/N)**2) + 0.016**2/N

#     return avg, np.sqrt(var) / np.sqrt(N)

class WiseAggregator:
    def __init__(
        self,
        band: Literal["W1", "W2"],
    ):
        self.band = band

    def aggregate(
        self,
        values: ArrayLike,
        errors: Optional[ArrayLike] = None
    ) -> Tuple[float, float]:

        band = self.band

        if band is None:
            raise ValueError(
                "`band` is required for WiseMeanAggregator: W1 or W2."
            )

        if band == 'W1':
            zp = W1.zp_vega
        elif band == 'W2':
            zp = W2.zp_vega
        else:
            raise ValueError(
                f"Unsupported WISE band: {band}"
            )

        values = np.asarray(values, dtype=float)

        if errors is not None:
            errors = np.asarray(errors, dtype=float)

        valid = np.isfinite(values)

        if errors is not None:
            valid &= np.isfinite(errors)

        values = values[valid]

        if errors is not None:
            errors = errors[valid]

        N = len(values)

        if N == 0:
            return np.nan, np.nan

        flux = mag2flux(values, zp)

        flux_avg = np.mean(flux)

        avg = flux2mag(flux_avg, zp)

        var = np.sum(
            (values - avg) ** 2
        ) / (N - 1)

        if errors is not None:
            var += np.sum(
                (errors / N) ** 2
            )

        var += 0.016 ** 2 / N

        return avg, np.sqrt(var / N)
    