"""Reusable, survey-agnostic light-curve processing nodes.

Nodes operate on a :class:`~easycat.pipeline.core.DataPacket` and should not
perform batch orchestration, checkpointing or output-directory policy.
"""

import numpy as np
import pandas as pd
import logging

from typing import (
    Optional,
    Dict,
    Any,
    Tuple,
    List
)

from numpy.typing import NDArray

from astropy.coordinates import SkyCoord
from astropy import units as u

from easycat.util import dbscan
from easycat.lightcurve.reprocess import (
    create_grouper,
    create_aggregator,
    create_outlier_detector,
    Aggregator
)

from .core import ProcessingNode, DataPacket

logger = logging.getLogger('easycat')

class PositionFilterNode(ProcessingNode):
    """
    Processing node for filtering light curves based on spatial position.
    
    Supports both DBSCAN clustering and cone cut methods and can be used for
    any telescope data with positional information.

    Processing order is:

    1. calculate the maximum separation before filtering;
    2. optionally retain the dominant DBSCAN cluster around ``pos_ref``;
    3. optionally apply a hard cone cut in arcseconds.

    The node records the number removed by each stage in ``data.results``.
    """
    def __init__(
        self,
        ra_column: str = 'ra',
        dec_column: str = 'dec',
        use_dbscan: bool = False,
        dbscan_radius: float = 0.5,
        min_neighbors: int = 5,
        min_cluster_size: int = 1,
        cone_radius: Optional[float] = None,
        name: str = "filter@position",
        **kwargs
    ):
        """
        Initialize position filter node.
        
        Parameters
        ----------
        name : str
            Node name
        ra_column : str
            Column name for Right Ascension (degrees)
        dec_column : str
            Column name for Declination (degrees)
        use_dbscan : bool
            Whether to apply DBSCAN clustering
        dbscan_config : Optional[Dict[str, Any]]
            Configuration for DBSCAN algorithm
        cone_radius : float
            Cone cut radius in arcseconds (if None, no cone cut)
        """
        super().__init__(name, **kwargs)

        self.config.update({
            'ra_column': ra_column,
            'dec_column': dec_column,
            'use_dbscan': use_dbscan,
            'dbscan_radius': dbscan_radius,
            'min_neighbors': min_neighbors,
            'min_cluster_size': min_cluster_size,
            'cone_radius': cone_radius
        })
        self.required_columns = (ra_column, dec_column)

    def validate(self, data: DataPacket) -> Tuple[bool, Optional[str]]:
        """Check position columns and the optional reference coordinate."""
        is_valid, error_msg = super().validate(data)
        if not is_valid:
            return False, error_msg

        lc = data.light_curve
        required_columns = [self.config['ra_column'], self.config['dec_column']]
        missing = [col for col in required_columns if col not in lc.columns]

        if missing:
            return False, f"Missing position columns: {missing}."

        if data.metadata.get("pos_ref") is None:
            return False, f"Missing reference position in metadata."

        return True, None

    def _calc_max_sep(self, lc: pd.DataFrame) -> float:
        """Return the largest angular offset from the median position."""
        ra_colname = self.config['ra_column']
        dec_colname = self.config['dec_column']

        ra = lc[ra_colname].to_numpy()
        dec = lc[dec_colname].to_numpy()

        coords = SkyCoord(ra, dec, frame="fk5", unit="deg")
        coord_median = SkyCoord(np.median(ra), np.median(dec), frame="fk5", unit="deg")

        sep = coords.separation(coord_median)
        sep = sep.to_value(u.arcsec)

        return np.max(sep)

    def process(self, data: DataPacket) -> DataPacket:
        """Apply the configured DBSCAN and/or cone-position cuts."""
        lc = data.light_curve.copy()
        initial_count = len(lc)

        # The reference position is runtime data, not node configuration; this
        # lets one Pipeline instance be reused for many sources.
        pos_ref = data.metadata.get("pos_ref")
        use_dbscan = self.config['use_dbscan']
        cone_radius = self.config['cone_radius']

        if (cone_radius is not None) or use_dbscan:
            if pos_ref is None:
                data.add_error("pos_ref is required for cone cut or DBSCAN filter.", self.name)
                return data

        if len(lc) > 0:
            data.add_result('sep_max', self._calc_max_sep(lc), self.name)
        else:
            data.add_result('sep_max', None, self.name)

        # DBSCAN removes additional detections of neighbouring objects in the
        # same WISE query rather than merely applying a circular cut.
        if use_dbscan:
            try:
                lc = dbscan.filter_dbscan(
                    lc,
                    pos_ref=pos_ref,
                    radius=self.config['dbscan_radius'] * u.arcsec,
                    min_neighbors=self.config['min_neighbors'],
                    min_cluster_size=self.config['min_cluster_size'],
                    ra_column=self.config['ra_column'],
                    dec_column=self.config['dec_column']
                )

                if len(lc) > 0:
                    data.add_result('sep_max_after_dbscan', self._calc_max_sep(lc), self.name)
                else:
                    data.add_result('sep_max_after_dbscan', None, self.name)

                data.add_result('dbscan_removed', initial_count - len(lc), self.name)
            except Exception as e:
                data.add_error(f"DBSCAN failed - {str(e)}", self.name)
                return data
        
        # Apply a hard reference-radius cut after clustering when requested.
        if cone_radius is not None:
            try:
                # Convert radius to appropriate units
                if isinstance(cone_radius, (int, float)):
                    cone_radius = cone_radius * u.arcsec

                # Create sky coordinates from data
                coords = SkyCoord(
                    lc[self.config['ra_column']], 
                    lc[self.config['dec_column']], 
                    frame="fk5", 
                    unit="deg"
                )
                
                # Calculate separations
                separations = coords.separation(pos_ref)
                
                # Apply mask
                mask = separations <= cone_radius
                
                # Apply filter
                before_cone = len(lc)
                lc: pd.DataFrame = lc[mask]
                lc.reset_index(drop=True, inplace=True)
                
                # Record statistics
                cone_removed = before_cone - len(lc)
                data.add_result('cone_removed', cone_removed, self.name)
                
            except Exception as e:
                data.add_error(f"Cone cut failed - {str(e)}", self.name)
                return data
        
        data.light_curve = lc
        return data


class BinningNode(ProcessingNode):
    """
    General node for binning time-series data.
    
    Groups data into time bins and aggregates values within each bin.  The
    grouping is computed once from ``time_column`` and shared by every value
    column.  Each output row contains the bin center, bin size, duration,
    aggregated values and propagated errors.
    """
    
    def __init__(
        self,
        name: str = "BinningNode",
        time_column: str = "mjd",
        value_columns: Optional[List[str]] = None,
        error_columns: Optional[Dict[str, str]] = None,
        group: Optional[Dict[str, Any]] = None,
        aggregate: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        super().__init__(name)
        
        self.config.update({
            'time_column': time_column,

            'value_columns': value_columns or ['flux', 'mag'],
            'error_columns': error_columns or {
                'flux': 'flux_err',
                'mag': 'mag_err'
            },

            'group': group or {
                'method': 'max_interval',
                'max_interval': 1.2
            },

            'aggregate': aggregate or {
                'flux': {
                    'method': 'mean',
                },
                'mag': {
                    'method': 'mean',
                },
            },
        })
        self.required_columns = (time_column,)

        self.grouper = None
        self.aggregators = {}
    
    def validate(self, data: DataPacket) -> Tuple[bool, Optional[str]]:
        """Validate columns and aggregation configuration without state changes."""

        is_valid, error_msg = super().validate(data)
        if not is_valid:
            return False, error_msg
        
        lc = data.light_curve
        
        # Check time column
        time_column = self.config["time_column"]

        if time_column not in lc.columns:
            return (
                False,
                f"Time column '{time_column}' not found"
            )
        
        # Check value columns
        value_columns = self.config["value_columns"]

        available_columns = [
            col
            for col in value_columns
            if col in lc.columns
        ]

        if not available_columns:
            return (
                False,
                f"No value columns found from {value_columns}"
            )

        missing_columns = [
            col
            for col in value_columns
            if col not in lc.columns
        ]

        if missing_columns:
            print(
                f"[{self.name}] "
                f"Missing value columns: {missing_columns}"
            )

        # Validate config factories here, but create fresh stateful instances
        # in process() so two sources/workers never share mutable helpers.
        try:
            create_grouper(self.config["group"])
        except (TypeError, ValueError) as exc:
            return False, str(exc)

        aggregate_config = self.config["aggregate"]

        for value_col in available_columns:

            if value_col not in aggregate_config:
                return (
                    False,
                    f"No aggregation configuration for "
                    f"value column '{value_col}'"
                )

            try:
                create_aggregator(
                    aggregate_config[value_col]
                )
            except (TypeError, ValueError) as exc:
                return (
                    False,
                    f"Failed to create aggregator for "
                    f"column '{value_col}': {exc}"
                )
        
        return True, None
    
    def process(self, data: DataPacket) -> DataPacket:
        """Group observations and aggregate each requested value column."""
        lc = data.light_curve
        grouper = create_grouper(self.config["group"])
        aggregators = {
            value_col: create_aggregator(self.config["aggregate"][value_col])
            for value_col in self.config["value_columns"]
            if value_col in lc.columns
        }

        time_column = self.config["time_column"]
        value_columns = self.config["value_columns"]
        error_columns = self.config["error_columns"]

        times: NDArray = lc[self.config['time_column']].to_numpy()

        # ---------------------------------------------------------
        # Group time points once.
        # All value columns share the same grouping.
        # ---------------------------------------------------------
        los, his = grouper.group(times)
        binned_data = []

        for lo, hi in zip(los, his):
            bin_indices = list(range(lo, hi + 1))
            bin_size = hi - lo + 1
            
            # Bin center is the median observation time; duration is the
            # within-bin span and is useful for interpreting WISE visits.
            bin_times = times[lo:hi+1]
            bin_center = np.median(bin_times)
            bin_duration = (
                bin_times[-1] - bin_times[0]
                if len(bin_times) > 1
                else 0
            )
            
            bin_record = {
                time_column: bin_center,
                'bin_size': bin_size,
                'duration': bin_duration
            }
            
            # Aggregate each band independently, sharing only the time bins.
            for value_col in value_columns:
                # Skip columns that are not available.
                if value_col not in lc.columns:
                    continue

                aggregator: Aggregator = aggregators[value_col]

                values: NDArray = lc.iloc[bin_indices][value_col].to_numpy()

                # Get errors if available
                error_col = error_columns.get(value_col)
                errors = None

                if (
                    error_col is not None
                    and error_col in lc.columns
                ):
                    errors = lc.iloc[bin_indices][error_col].to_numpy()

                # Aggregate
                binned_value, binned_error = (
                    aggregator.aggregate(
                        values,
                        errors,
                    )
                )

                # Add to bin record
                bin_record[value_col] = binned_value
                if error_col:
                    bin_record[error_col] = binned_error
            
            # Add bin record to list
            binned_data.append(bin_record)
            
        
        # Create binned DataFrame
        if binned_data:
            binned_df = pd.DataFrame(binned_data)
            
            # Ensure proper column order
            bin_cols = [time_column, 'bin_size', 'duration']
            columns_order = bin_cols + \
                           [col for col in binned_df.columns 
                            if col not in bin_cols]
            binned_df = binned_df[columns_order]
        else:
            binned_df = pd.DataFrame()

        data.light_curve = binned_df
        data.add_result("input_rows", len(lc), self.name)
        data.add_result("output_rows", len(binned_df), self.name)
        return data


class OutlierFilterNode(ProcessingNode):
    """
    Processing node for detecting and removing outliers in light curves.
    
    The processing consists of two steps:

    1. group data points according to a time-grouping method;
    2. detect outliers independently within each group and band.

    A point is removed if any requested value column marks it as an outlier.

    Configuration
    -------------
    grouping:
        Configuration for grouping data by time.

    outlier:
        Configuration for outlier detection.
        
        Supported methods: ["mad", "iqr"]
    """
    def __init__(
        self,
        name: str = "filter@outlier",
        time_column: str = "mjd",
        value_columns: Optional[List[str]] = None,
        group: Optional[Dict[str, Any]] = None,
        outlier: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        name : str
            Node name.

        time_column : str
            Column name for time values.

        value_columns : Optional[List[str]]
            Columns in which outliers should be detected.
            Defaults to ['flux', 'mag'].

        outlier_threshold : float
            Sigma threshold for outlier detection

        group : Optional[Dict[str, Any]]
            Configuration for time grouping.

        outlier : Optional[Dict[str, Any]]
            Configuration for outlier detection.
        
        **kwargs
            Additional arguments passed to ProcessingNode.
        """
        super().__init__(name, **kwargs)
        
        self.config.update({
            'time_column': time_column,
            'value_columns': value_columns or ['flux', 'mag'],

            'group': group
            or {
                'method': 'max_interval',
                'max_interval': 1.2,
            },

            'outlier': outlier
            or {
                'method': 'sigma_clip',
                'threshold': 5.0,
            },
        })
        self.required_columns = (time_column,)
    
    def validate(self, data: DataPacket) -> Tuple[bool, Optional[str]]:
        """Validate columns and detector configuration without state changes."""

        is_valid, error_msg = super().validate(data)
        
        if not is_valid:
            return False, error_msg
        
        lc = data.light_curve

        # Check time column
        time_column = self.config["time_column"]

        if time_column not in lc.columns:
            return False, f"Missing time column: '{time_column}'"

        # Check value columns
        value_columns = self.config["value_columns"]

        available_columns = [
            col for col in value_columns
            if col in lc.columns
        ]

        if not available_columns:
            return (
                False,
                f"No available value columns from {value_columns}",
            )

        # Some value columns are missing, but this is not fatal.
        if len(available_columns) < len(value_columns):
            missing_columns = [
                col for col in value_columns
                if col not in lc.columns
            ]

            logger.warning(
                "Some value columns are missing for %s: %s. Using available columns.",
                self.name,
                missing_columns
            )

        # Construct and discard helpers here to validate user configuration;
        # process() creates the per-execution instances actually used below.
        try:
            create_outlier_detector(self.config["outlier"])
            create_grouper(self.config["group"])
        except (TypeError, ValueError) as exc:
            return False, str(exc)

        return True, None
    
    def process(self, data: DataPacket) -> DataPacket:
        """Detect and remove outliers from the light curve."""
        
        lc = data.light_curve.copy()
        self.detector = create_outlier_detector(self.config["outlier"])
        self.grouper = create_grouper(self.config["group"])
        # initial_count = len(lc)
        
        # Group first so cadence-dependent noise is compared only within a
        # physically similar observing window.
        times: NDArray = lc[self.config['time_column']].to_numpy()
        
        # Group data by time
        los, his = self.grouper.group(times)
        
        # Outlier decisions are made independently per band.  The union is
        # removed because one obviously bad band measurement contaminates the
        # resulting colour/magnitude record.
        outlier_indices = set()
        value_columns = self.config["value_columns"]

        for lo, hi in zip(los, his):
            group_indices = list(range(lo, hi + 1))
            group_data = lc.iloc[group_indices]

            # Check each value column independently.
            for value_column in value_columns:
                if value_column not in group_data.columns:
                    continue

                values: NDArray = group_data[value_column].to_numpy()

                # Skip if all values are NaN.
                if np.all(np.isnan(values)):
                    continue

                outliers_in_column = self.detector.detect(values)

                # Convert local group indices to global
                # light-curve indices.
                for i, is_outlier in enumerate(outliers_in_column):
                    if is_outlier:
                        outlier_indices.add(
                            group_indices[i]
                        )
        
        # Remove outliers
        outlier_indices = sorted(outlier_indices)
        
        if outlier_indices:
            # Convert to set for faster lookup
            outlier_set = set(outlier_indices)
            keep_mask = np.array(
                [i not in outlier_set for i in range(len(lc))],
                dtype=bool
            )
            
            lc_filtered = lc[keep_mask].copy()
            lc_filtered.reset_index(drop=True, inplace=True)
            
            # Record statistics
            outliers_removed = len(outlier_indices)
            data.add_result('removed', outliers_removed, self.name)
            
            # Add outlier information to data packet
            outlier_info = {
                'indices': outlier_indices,
                'times': times[outlier_indices].tolist() if len(outlier_indices) > 0 else [],
                'count': outliers_removed
            }
            data.add_result('outlier_info', outlier_info, self.name)
        else:
            lc_filtered = lc
            data.add_result('removed', 0, self.name)
        
        data.light_curve = lc_filtered
        data.add_result("input_rows", len(lc), self.name)
        data.add_result("output_rows", len(lc_filtered), self.name)
        return data


class EpochCleanNode(ProcessingNode):
    """Remove observing epochs with too few valid points.

    This node is intended for binned or repeated-exposure data where a single
    epoch should contain several measurements.  It does not remove individual
    points; it keeps or drops an entire time group.
    """
    def __init__(
        self,
        name: str = "EpochClean",
        time_column: str = "mjd",
        group: Optional[Dict[str, Any]] = None,
        min_points_per_epoch: int = 5,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.config.update({
            'time_column': time_column,
            'group': group
            or {
                'method': 'max_interval',
                'max_interval': 1.2,
            },
            'min_points_per_epoch': min_points_per_epoch
        })
        self.required_columns = (time_column,)
    
    def validate(self, data: DataPacket) -> Tuple[bool, Optional[str]]:
        """Validate the time column and grouping configuration."""
        is_valid, error_msg = super().validate(data)
        if not is_valid:
            return False, error_msg
        
        lc = data.light_curve
        time_column = self.config['time_column']
        
        if time_column not in lc.columns:
            return False, f"Time column '{time_column}' not found"

        # Check that the configured grouper can be constructed without
        # retaining mutable state on the node.
        try:
            create_grouper(self.config["group"])
        except (TypeError, ValueError) as exc:
            return False, str(exc)
        
        return True, None
    
    def process(self, data: DataPacket) -> DataPacket:
        """Drop epochs below ``min_points_per_epoch``."""
        lc = data.light_curve.copy()
        self.grouper = create_grouper(self.config["group"])
        initial_count = len(lc)

        time_column = self.config['time_column']
        min_points_per_epoch = self.config['min_points_per_epoch']
        
        times = lc[time_column].to_numpy()
        los, his = self.grouper.group(times)
        
        # Work at epoch level so a small visit is either retained as a whole
        # or rejected as a whole.
        keep_mask = np.full(len(lc), False, dtype=bool)
        epochs_removed = 0
        
        for lo, hi in zip(los, his):
            epoch_size = hi - lo + 1
            
            if epoch_size >= min_points_per_epoch:
                # Keep this epoch
                keep_mask[lo:hi+1] = True
            else:
                # Remove this epoch
                epochs_removed += 1
        
        # Apply mask
        lc = lc[keep_mask]
        lc.reset_index(drop=True, inplace=True)
        
        # Record statistics
        total_removed = initial_count - len(lc)
        
        data.add_result('epochs_total', len(los), self.name)
        data.add_result('epochs_removed', epochs_removed, self.name)
        data.add_result('points_removed', total_removed, self.name)
        data.add_result("input_rows", initial_count, self.name)
        data.add_result("output_rows", len(lc), self.name)
        
        data.light_curve = lc
        return data
