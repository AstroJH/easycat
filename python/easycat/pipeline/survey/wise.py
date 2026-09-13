"""WISE-specific light-curve loading, quality cuts, analysis and storage."""
from pathlib import Path
import json
import os
import tempfile

from typing import Union, Tuple, Literal, Optional
from numpy.typing import ArrayLike

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from astropy.io import fits
from astropy.table import Table

from easycat.astrofilter import FilterDB
from easycat.util.photometry import mag2flux, flux2mag
from easycat.pipeline import DataPacket, ProcessingNode

_FILTER_DB = FilterDB()
W1 = _FILTER_DB.get('WISE/WISE.W1')
W2 = _FILTER_DB.get('WISE/WISE.W2')


def _decode_series(values):
    """Decode bytes and normalize WISE flag strings."""
    out = []
    for value in values:
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("ascii", errors="replace")
        if value is None:
            out.append("")
        else:
            out.append(str(value).strip())
    return pd.Series(out, index=getattr(values, "index", None))


def _normalise_wise_flags(lc: pd.DataFrame) -> pd.DataFrame:
    out = lc.copy()
    for column in ("cc_flags", "moon_masked", "var_flg", "ph_qual"):
        if column in out.columns:
            out[column] = _decode_series(out[column])
    return out


class WiseLcLoader(ProcessingNode):
    """Load one per-source WISE FITS product into a DataPacket.

    WISE FITS string columns are frequently returned as bytes by pandas; the
    loader normalizes the known flag columns immediately so downstream nodes
    can compare them as ordinary strings.
    """
    def __init__(self, name: str = 'loader@wise'):
        super().__init__(name, minsize=0, required_metadata=("filepath",))

    def process(self, data: DataPacket) -> DataPacket:
        """Read HDU 1 and normalize its flag columns."""
        filepath = data.metadata.get('filepath')

        with fits.open(filepath) as hdul:
            lcurve = Table(hdul[1].data).to_pandas()

        data.light_curve = _normalise_wise_flags(lcurve)
        data.provenance.setdefault("input_file", str(Path(filepath).resolve()))
        return data


class WiseLcStorage(ProcessingNode):
    """Atomically persist a processed WISE light curve.

    FITS output records object identity, input path and a compact provenance
    JSON string in the primary header.  Both CSV and FITS paths are written
    through a temporary file followed by ``os.replace``.
    """
    def __init__(
        self,
        name: str = 'storage@wise',
        fmt: Literal['csv', 'fits'] = 'fits'
    ):
        super().__init__(name, minsize=0, required_metadata=("storage_path",))

        self.config['format'] = fmt

    def process(self, data: DataPacket) -> DataPacket:
        """Write the current light curve to ``metadata['storage_path']``."""
        fmt = self.config.get('format')
        output = Path(data.metadata['storage_path'])

        lcurve = data.light_curve

        if lcurve is None or len(lcurve) <= 0:
            data.add_warning(
                'empty data', self.name
            )
            return data

        # All output is written beside the final file so os.replace() remains
        # atomic on the same filesystem.
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=str(output.parent), prefix=output.name + ".", suffix=".tmp",
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            if fmt == 'csv':
                lcurve.to_csv(temp_path, index=False)
            elif fmt == 'fits':
                table = Table.from_pandas(lcurve, index=False, units=None)
                primary_hdu = fits.PrimaryHDU()
                primary_hdu.header['PIPELINE'] = 'wise_standard'
                primary_hdu.header['OBJID'] = str(data.obj_id or '')
                primary_hdu.header['INFILE'] = str(
                    data.provenance.get('input_file', data.metadata.get('filepath', ''))
                )[:68]
                primary_hdu.header['PROV'] = json.dumps(
                    data.provenance, default=str, ensure_ascii=False,
                )
                table_hdu = fits.BinTableHDU(table, name='WISELC')
                table_hdu.header['EXTNAME'] = 'WISELC'
                table_hdu.header['AUTHOR'] = 'Virjid'
                table_hdu.header['CREATED'] = pd.Timestamp.now().isoformat()
                fits.HDUList([primary_hdu, table_hdu]).writeto(
                    temp_path, overwrite=True,
                )
            else:
                raise ValueError("Supported format: csv and fits")
            os.replace(temp_path, output)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        data.artifacts['storage_path'] = output
        data.add_result('storage_path', str(output), self.name)

        return data


class WisePreprocessNode(ProcessingNode):
    """Normalize raw WISE rows before applying scientific quality cuts.

    Responsibilities are deliberately limited to type/value sanitation and
    sorting: bytes flags are decoded, required numeric fields are coerced,
    missing values are removed and rows are sorted by MJD.
    """
    MISSING_CHECK_FIELDS = ["mjd", "w1mag", "w2mag", "w1sigmag", "w2sigmag", "na", "nb"]

    def __init__(
        self,
        name: str = 'preprocess@wise',
        missing_value: float = -1
    ):
        super().__init__(
            name,
            required_columns=(
                "mjd", "w1mag", "w2mag", "w1sigmag", "w2sigmag",
            ),
        )
        self.config['sort_column'] = 'mjd'
        self.config['missing_value'] = missing_value

    def process(self, data: DataPacket) -> DataPacket:
        """Remove invalid/missing rows and sort the light curve by time."""
        lc = _normalise_wise_flags(data.light_curve)
        initial_count = len(lc)
        # A row is invalid when any required numeric quantity is null, NaN or
        # equal to the archive missing sentinel (-1 by default).  WISE can mix
        # masked columns and ordinary NaNs, so both paths are handled here.
        mask = pd.Series(True, index=lc.index)
        for field in WisePreprocessNode.MISSING_CHECK_FIELDS:
            if field not in lc.columns:
                continue
            values = pd.to_numeric(lc[field], errors="coerce")
            mask &= values.notna()
            mask &= values != self.config['missing_value']

        lc = lc[mask].copy()
        for field in WisePreprocessNode.MISSING_CHECK_FIELDS:
            if field in lc.columns:
                lc[field] = pd.to_numeric(lc[field], errors="coerce")
        lc.sort_values(by=self.config['sort_column'], inplace=True)
        lc.reset_index(drop=True, inplace=True)

        data.light_curve = lc
        data.add_result('input_rows', initial_count, self.name)
        data.add_result('output_rows', len(lc), self.name)
        data.add_result('removed_invalid', initial_count - len(lc), self.name)

        # Record statistics
        data.metadata['sorted'] = True
        return data


class WiseBasicCriteriaNode(ProcessingNode):
    """Apply the basic per-epoch WISE quality criteria.

    Each criterion is evaluated separately and recorded as a cut-flow metric.
    This node is for multi-epoch photometry; All-Sky source-catalog flags such
    as ``ext_flg`` are not treated as per-epoch criteria.
    """

    def __init__(
        self,
        name: str = "basic_criteria@wise",
        max_nb: int = 2,
        max_w1rchi2: float = 5,
        max_w2rchi2: float = 5
    ):
        super().__init__(
            name,
            required_columns=(
                "na", "nb", "saa_sep", "qi_fact", "qual_frame",
                "w1rchi2", "w2rchi2", "moon_masked", "cc_flags",
            ),
        )

        self.config.update({
            'max_nb': max_nb,
            'max_w1rchi2': max_w1rchi2,
            'max_w2rchi2': max_w2rchi2
        })

    def process(self, data: DataPacket) -> DataPacket:
        """Evaluate all quality criteria and retain rows passing every cut."""

        lc = _normalise_wise_flags(data.light_curve).copy()
        initial_count = len(lc)

        max_nb = self.config['max_nb']
        max_w1rchi2 = self.config['max_w1rchi2']
        max_w2rchi2 = self.config['max_w2rchi2']

        # Define conditions based on WISE quality flags
        na = pd.to_numeric(lc["na"], errors="coerce")
        nb = pd.to_numeric(lc["nb"], errors="coerce")
        saa_sep = pd.to_numeric(lc["saa_sep"], errors="coerce")
        qi_fact = pd.to_numeric(lc["qi_fact"], errors="coerce")
        qual_frame = pd.to_numeric(lc["qual_frame"], errors="coerce")
        w1rchi2 = pd.to_numeric(lc["w1rchi2"], errors="coerce")
        w2rchi2 = pd.to_numeric(lc["w2rchi2"], errors="coerce")

        # Keep conditions separate so the notebook can report exactly which
        # quality criterion rejected each epoch; the final mask is their AND.
        conditions = {
            'quality': (((qual_frame > 0) | (qual_frame == -1)) & (qi_fact == 1)).fillna(False),
            'coverage': ((na == 0) & (nb <= max_nb)).fillna(False),
            'saa': (saa_sep > 0).fillna(False),
            'moon': (lc['moon_masked'].astype(str).str[:2] == '00').fillna(False),
            'cc': (lc['cc_flags'].astype(str).str[:2] == '00').fillna(False),
            'rchi2': ((w1rchi2 <= max_w1rchi2) & (w2rchi2 <= max_w2rchi2)).fillna(False),
        }

        # Apply all conditions.
        mask = pd.Series(True, index=lc.index)
        for condition in conditions.values():
            mask &= condition
        lc_filtered = lc[mask].copy()
        lc_filtered.reset_index(drop=True, inplace=True)

        # Record statistics
        removed = len(lc) - len(lc_filtered)
        data.add_result('removed', removed, self.name)
        data.add_result('input_rows', initial_count, self.name)
        data.add_result('output_rows', len(lc_filtered), self.name)
        for label, condition in conditions.items():
            data.add_result(
                f'removed_{label}', int((~condition).sum()), self.name,
            )

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

        super().__init__(
            name,
            required_columns=(
                "mjd", "w1mag", "w2mag", "w1sigmag", "w2sigmag",
            ),
        )

        self.config.update({
            'x_years': x_years,
            'weighted': weighted
        })

    def process(self, data: DataPacket) -> DataPacket:
        """Calculate local W1/W2 magnitude slopes around each observation."""

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
        # The slope at each epoch is fit from the window t0 +/- x_years; it is
        # a local variability diagnostic, not a single global trend.
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
        data.add_result('slope_valid_w1', int(np.isfinite(k_w1).sum()), self.name)
        data.add_result('slope_valid_w2', int(np.isfinite(k_w2).sum()), self.name)

        return data


class WiseAnalyzer(ProcessingNode):
    """Compute standard WISE colour-variability summary metrics."""
    def __init__(self, name: str = 'analyzer', minsize: int = 1, N: int = 1000):
        ProcessingNode.__init__(
            self, name, minsize,
            required_columns=(
                "mjd", "w1mag", "w2mag", "w1sigmag", "w2sigmag",
            ),
        )
        self.N = N
        self.sigma_probs = {
            1: 0.6826895,  # 1σ: 68.27%
            2: 0.9544997,  # 2σ: 95.45%
            3: 0.9973002   # 3σ: 99.73%
        }

    def process(self, data: DataPacket) -> DataPacket:
        """Measure W1/W2 colour correlations, extrema and variability errors."""
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

        # w1varamp = intrinsic_variability_amplitude(w1mag, w1err, False) # * np.sqrt(1+row.Z)
        # w2varamp = intrinsic_variability_amplitude(w2mag, w2err, False) # * np.sqrt(1+row.Z)
        w1varerr = np.sqrt(np.mean(w1err**2))
        w2varerr = np.sqrt(np.mean(w2err**2))
        # data.add_result(key='w1varamp', value=w1varamp, node_name=self.name)
        # data.add_result(key='w2varamp', value=w2varamp, node_name=self.name)
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


class WiseAggregator:
    """Aggregate repeated WISE observations in flux space.

    Magnitudes are converted to flux before averaging, which avoids the bias
    introduced by directly averaging logarithmic magnitudes.
    """
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
        """Return the flux-averaged magnitude and its epoch uncertainty.

        The formal variance follows the analysis prescription

        ``Var(m_epoch) =
        sum((m_i - m_epoch)^2) / (N (N - 1))
        + sum(error_i^2) / N^2
        + sigma_sys^2 / N``.

        The first term is omitted for ``N == 1`` because the scatter variance
        is undefined for a single measurement.
        """

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
            # Measurement errors are standard deviations, so non-negative
            # finite values are the only physically meaningful inputs.
            valid &= np.isfinite(errors) & (errors >= 0)

        values = values[valid]

        if errors is not None:
            errors = errors[valid]

        N = len(values)

        if N == 0:
            return np.nan, np.nan

        flux = mag2flux(values, zp)

        flux_avg = np.mean(flux)

        avg = flux2mag(flux_avg, zp)

        # Scatter term.  It is only defined for at least two measurements.
        variance = 0.0
        if N > 1:
            variance += np.sum((values - avg) ** 2) / (N * (N - 1))

        # Propagated per-exposure measurement uncertainty.
        if errors is not None:
            variance += np.sum(errors**2) / N**2

        # Adopted systematic uncertainty, reduced by the number of exposures.
        variance += 0.016**2 / N

        return avg, np.sqrt(max(float(variance), 0.0))
