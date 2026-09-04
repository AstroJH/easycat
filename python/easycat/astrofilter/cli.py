"""``easycat filter ...`` subcommand implementation."""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import astropy.units as u

from .common import search_common
from .db import FilterDB

JSON_KEYS = [
    "filterID", "facility", "instrument", "band", "detector_type",
    "n_points", "wavelength_range_AA",
    "wl_pivot_AA", "wl_pivot_svo_AA",
    "wl_mean_AA", "wl_mean_svo_AA",
    "wl_eff_svo_AA",
    "fwhm_AA", "fwhm_svo_AA",
    "zp_vega_Jy", "zp_vega_svo_Jy",
    "zp_ab_Jy", "mag_sys",
]


def _filter_info(filt) -> Dict[str, Any]:
    meta = filt.metadata
    info: Dict[str, Any] = {
        "filterID": filt.filter_id,
        "facility": filt.facility,
        "instrument": filt.instrument,
        "band": filt.band,
        "detector_type": str(filt.detector_type),
        "n_points": len(filt.wavelength),
        "wavelength_range_AA": [
            round(filt.wl_min.to_value(u.AA), 1),
            round(filt.wl_max.to_value(u.AA), 1),
        ],
        "wl_pivot_AA": round(filt.wl_pivot.to_value(u.AA), 2),
        "wl_mean_AA": round(filt.wl_mean.to_value(u.AA), 2),
        "fwhm_AA": round(filt.fwhm.to_value(u.AA), 2),
        "zp_ab_Jy": round(filt.zp_ab.to_value(u.Jy), 4),
    }
    # SVO-provided values for comparison / validation
    for key, out in [
        ("WavelengthPivot", "wl_pivot_svo_AA"),
        ("WavelengthMean", "wl_mean_svo_AA"),
        ("WavelengthEff", "wl_eff_svo_AA"),
        ("FWHM", "fwhm_svo_AA"),
        ("ZeroPoint", "zp_vega_svo_Jy"),
        ("MagSys", "mag_sys"),
        ("PhotSystem", "phot_system"),
        ("PhotCalID", "phot_cal_id"),
        ("Description", "description"),
    ]:
        val = meta.get(key)
        if val is not None:
            info[out] = val
    # Vega zero point: use the *persisted* local value when available
    # (never trigger a network fetch here); otherwise fall back to the SVO
    # metadata stored with the filter.
    if filt._zp_vega_jy is not None:
        info["zp_vega_Jy"] = round(filt._zp_vega_jy.to_value(u.Jy), 4)
        info["zp_vega_source"] = "computed"
    elif meta.get("ZeroPoint") is not None:
        try:
            info["zp_vega_Jy"] = round(float(meta["ZeroPoint"]), 4)
            info["zp_vega_source"] = "svo_metadata"
        except (TypeError, ValueError):
            pass
    info["svo_metadata"] = {k: v for k, v in meta.items()}
    return info


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def cmd_info(args: argparse.Namespace) -> int:
    db = FilterDB(cache_dir=args.cache)
    if args.refresh:
        # online: re-fetch and (re)compute + persist the local Vega zero point
        filt = db.fetch(args.filter, refresh=True)
    else:
        filt = db.get(args.filter)   # memory -> local cache only (offline)

    if filt is None:
        print(f"filter not found: {args.filter}")
        return 1

    info = _filter_info(filt)
    if args.json:
        _print_json(info)
        return 0

    print(f"filterID:     {filt.filter_id}")
    print(f"facility:     {filt.facility}")
    print(f"instrument:   {filt.instrument}")
    print(f"band:         {filt.band}")
    print(f"detector:     {filt.detector_type} "
          f"(SVO DetectorType={filt.metadata.get('DetectorType')})")
    print(f"n_points:     {len(filt.wavelength)}")
    print(f"range:        {filt.wl_min:.1f} - {filt.wl_max:.1f}")
    print(f"pivot (ours): {info['wl_pivot_AA']} AA"
          + (f"  | SVO: {info['wl_pivot_svo_AA']}" if "wl_pivot_svo_AA" in info else ""))
    print(f"mean  (ours): {info['wl_mean_AA']} AA"
          + (f"  | SVO: {info['wl_mean_svo_AA']}" if "wl_mean_svo_AA" in info else ""))

    if "wl_eff_svo_AA" in info:
        print(f"eff   (SVO):  {info['wl_eff_svo_AA']} AA")

    print(f"FWHM  (ours): {info['fwhm_AA']} AA"
          + (f"  | SVO: {info['fwhm_svo_AA']}" if "fwhm_svo_AA" in info else ""))

    if "zp_vega_Jy" in info:
        if info.get("zp_vega_source") == "computed":
            label = "Vega ZP (ours)"
        else:
            label = "Vega ZP (SVO)"
        print(f"{label}: {info['zp_vega_Jy']} Jy"
              + (f"  | SVO: {info['zp_vega_svo_Jy']}"
                 if info.get("zp_vega_source") == "computed" and "zp_vega_svo_Jy" in info
                 else ""))
        if info.get("zp_vega_source") != "computed":
            print("       (From SVO metadata; run `easycat filter info <id> --refresh` "
                  "to compute and cache the local value online)")
    print(f"AB ZP:        {info['zp_ab_Jy']} Jy")

    if "description" in info:
        print(f"description:  {info['description']}")

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    db = FilterDB(cache_dir=args.cache)
    results: List[Dict[str, Any]] = []
    # 1) curated common list
    results.extend(search_common(args.query))
    # 2) local cache
    for r in db.search(args.query):
        r.setdefault("source", "cache")
        results.append(r)
    # 3) online (best effort)
    if not args.local_only:
        try:
            for r in db.search(args.query):
                if r.get("filterID") not in {x.get("filterID") for x in results}:
                    r["source"] = "svo2"
                    results.append(r)
        except Exception:
            pass

    seen = set()
    out = []
    for r in results:
        key = r.get("filterID")
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    if args.json:
        _print_json(out)
        return 0
    if not out:
        print(f"no filters match {args.query!r}")
        return 1
    for r in out:
        desc = r.get("Description") or r.get("description") or ""
        print(f"{r.get('filterID'):<28} [{r.get('source', '?')}] {desc}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib

    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    db = FilterDB(cache_dir=args.cache)
    filt = db.get(args.filter)
    if filt is None:
        print(f"filter not found: {args.filter}")
        return 1
    ax = filt.plot(normalized=not args.raw)
    if args.output:
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(args.output, dpi=150)
        print(f"saved to {args.output}")
    else:
        plt.show()
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache", help="filter cache directory")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable JSON output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="easycat filter")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_info = sub.add_parser("info", help="show filter details (offline)")
    p_info.add_argument("filter", help="filter ID, e.g. SLOAN/SDSS.r")
    p_info.add_argument("--refresh", action="store_true",
                        help="re-fetch from SVO2 and recompute/cache the "
                             "local Vega zero point")
    _add_common(p_info)
    p_info.set_defaults(func=cmd_info)

    p_search = sub.add_parser("search", help="search filters")
    p_search.add_argument("query", help="search text")
    p_search.add_argument("--local-only", action="store_true",
                          help="only search cached/common filters")
    _add_common(p_search)
    p_search.set_defaults(func=cmd_search)

    p_plot = sub.add_parser("plot", help="plot the transmission curve")
    p_plot.add_argument("filter", help="filter ID")
    p_plot.add_argument("-o", "--output", help="save plot to file")
    p_plot.add_argument("--raw", action="store_true",
                        help="plot the original (unnormalised) response")
    _add_common(p_plot)
    p_plot.set_defaults(func=cmd_plot)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
