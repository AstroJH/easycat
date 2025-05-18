from astropy.io import fits
from astropy.table import Table
import numpy as np


def notice_band(rmf_path, lo_kev, hi_kev):
    with fits.open(rmf_path) as hdul:
        ebounds = hdul["EBOUNDS"].data
        energy = (ebounds["E_MIN"] + ebounds["E_MAX"])/2


        chans = ebounds["CHANNEL"]
    
    mask = (energy >= lo_kev) & (energy <= hi_kev)
    notice_chans = chans[mask]
    return notice_chans[0], notice_chans[-1]


def get_head_and_tail(data_chan, data_cts):
    chans = data_chan[data_cts>0]

    if len(chans) <= 0:
        # TODO
        pass # no events

    return chans[0], chans[-1]


def skip_zerocts(data_cts, curr_lchan_ptr):
    size = len(data_cts)

    if curr_lchan_ptr >= size:
        return curr_lchan_ptr

    result = curr_lchan_ptr
    for i in range(curr_lchan_ptr, size):
        if data_cts[i] == 0:
            result += 1
        else:
            break
    
    return result


def group_min(data_chan, data_cts, mincts=1):
    size = len(data_chan)

    left_edges = []

    counter = 0
    bin_lptr = 0
    for i in range(0, size):
        counter += data_cts[i]

        if counter >= mincts:
            left_edges.append(data_chan[bin_lptr])

            # reset counter and update bin_lptr
            counter = 0
            bin_lptr = i + 1
            # bin_lptr = skip_zerocts(data_cts, bin_lptr)
    
    # if bin_lptr < size:
        # left_edges.append(data_chan[bin_lptr])
    bad_chan = data_chan[bin_lptr:] if bin_lptr < size else np.empty(0)
    
    return left_edges, bad_chan


def grppha(fn_spec, fn_output, mincts, notice=(), *, netreg:bool=False, fn_rmf=None, fn_spec_bkg=None, overwrite=False):
    """
    a simple group tool

    Parameters
    ----------
    fn_spec : str
        Pha filename of source spectrum.
    fn_output : str
        Output filename.
    mincts : number
        The grouping is set such that each new grouping contains a minimum of `mincts` counts in each bin.
        Channels that are defined as BAD are not included.
        Any spare channels at the end of the data are defined as BAD (QUALITY=2).
    notice : tuple | list
        The range of energy band noticed in keV, e.g., (0.3, 10.0).
    netreg : bool
        When `True`, **net** counts regulation will be used.
    fn_rmf : str, None
        Response filename of source spectrum.
        `fn_rmf` will be used only when `notice` is valid.
        If `fn_rmf` is `None`, the filename of response file will be obtained from fits header ("RESPFILE").
    fn_spec_bkg : str, None
        Pha filename of background spectrum.
        `fn_spec_bkg` will be used only when `netreg` is `True`.
        If `fn_spec_bkg` is `None`, the filename of background spectrum will be obtained from fits header ("BACKFILE").
    overwrite : bool
        If `True`, `overwrite` the output file if it exists.
        Raises an `OSError` if `False` and the output file exists.
        Default is False.
    """

    with fits.open(fn_spec) as hdul:
        hdu = hdul["SPECTRUM"]
        spec = hdu.data
        backscal = hdu.header["BACKSCAL"]
        exposure = hdu.header["EXPOSURE"]
        
        if fn_spec_bkg is None:
            fn_spec_bkg = hdu.header["BACKFILE"]
        
        if fn_rmf is None:
            fn_rmf = hdu.header["RESPFILE"]
    
    channel  = spec["CHANNEL"]
    spec_cts = spec["COUNTS"]
    
    grpflags = np.full(len(channel), fill_value=-1, dtype=np.int16)
    quaflags = np.full(len(channel), fill_value=0, dtype=np.int16)

    # Set notice band
    if len(notice) != 2:
        notice_chan_lo = channel[0]
        notice_chan_hi = channel[-1]
    else:
        notice_chan_lo, notice_chan_hi = notice_band(fn_rmf, notice[0], notice[1])
    
    # Set the channel with no photon count at the head and tail to BAD (QUALITY=5).
    #
    # XXX Note: Background photo counts are not considered.
    # The **net** counts may be <= 0 at the head and tail.
    head_chan, tail_chan = get_head_and_tail(channel, spec_cts)


    mask = (channel >= notice_chan_lo) & (channel <= notice_chan_hi) & (channel >= head_chan) & (channel <= tail_chan)
    quaflags[~mask] = 5
    grpflags[~mask] = 1

    data_chan = channel[mask]
    
    if netreg:
        with fits.open(fn_spec_bkg) as hdul:
            hdu = hdul["SPECTRUM"]
            spec_bkg = hdu.data
            backscal_bkg = hdu.header["BACKSCAL"]
            exposure_bkg = hdu.header["EXPOSURE"]

            area_ratio = backscal/backscal_bkg
            expo_ratio = exposure/exposure_bkg

        assert np.all(channel == spec_bkg["CHANNEL"])
        spec_bkg_cts = spec_bkg["COUNTS"]
        data_cts = spec_cts[mask]-spec_bkg_cts[mask]*area_ratio*expo_ratio
    else:
        data_cts = spec_cts[mask]
    
    left_edges, bad_chan = group_min(data_chan, data_cts, mincts)
    
    for i in range(0, len(channel)):
        if channel[i] in left_edges:
            grpflags[i] = 1
        
        if channel[i] in bad_chan: # spare channels at the end of the data
            grpflags[i] = 1
            quaflags[i] = 2

    
    # save the result to fn_output
    # XXX 
    with fits.open(fn_spec) as hdul:
        data = Table(hdul["SPECTRUM"].data)

        if "QUALITY" not in data.colnames:
            data.add_column(quaflags, name="QUALITY")
        else:
            data["QUALITY"] = quaflags
        
        if "GROUPING" not in data.colnames:
            data.add_column(grpflags, name="GROUPING")
        else:
            data["GROUPING"] = grpflags
        # data.add_columns(cols=[quaflags, grpflags], names=["QUALITY", "GROUPING"])

        new_hdul = []
        for hdu in hdul:
            if hdu.name != "SPECTRUM":
                new_hdul.append(hdu)
        
        hdu = fits.BinTableHDU(data,hdul["SPECTRUM"].header, name="SPECTRUM")
        hdu.header.update({
            "TFIELDS": len(data.colnames)
        })
        new_hdul.insert(1, hdu)
        fits.HDUList(new_hdul).writeto(fn_output, overwrite=overwrite)
