import numpy as np
from typing import Literal, Any

from astropy import units as u
from astropy.units import Quantity
from astropy import constants as const

FLUX_ZERO_DICT = {
    "W1": 309.540 * u.Jy,
    "W2": 171.787 * u.Jy,
    "J": 1594 * u.Jy,
    "H": 1024 * u.Jy,
    "K": 666.7 * u.Jy,
    "AB": 3631 * u.Jy
}

DEFAULT_UNIT_FLUX_WAVELENGTH = u.erg / u.cm / u.cm / u.s / u.AA
DEFAULT_UNIT_FLUX_FREQUENCE = u.Jy
DEFAULT_UNIT_WAVELENGTH = u.AA
DEFAULT_UNIT_FREQUENCE = u.Hz


def mag2flux(mag: float, zero_point: Quantity) -> Quantity:
    flux = zero_point / (10**(0.4*mag))
    return flux


def flux2mag(flux: Quantity, zero_point: Quantity) -> float:
    mag = 2.5 * np.log10(zero_point/flux)
    return mag.to_value()


def lamb2nu(
    wavelength: Quantity,
    unit: Any = DEFAULT_UNIT_FREQUENCE
) -> Quantity:
    return (const.c / wavelength).to(unit)


def nu2lamb(
    frequence: Quantity,
    unit: Any = DEFAULT_UNIT_WAVELENGTH
) -> Quantity:
    return (const.c / frequence).to(unit)


def flux_lamb2nu(
    flux_lamb: Quantity,
    wavelength: Quantity,
    unit: Any = DEFAULT_UNIT_FLUX_FREQUENCE
) -> Quantity:
    flux_nu: Quantity = wavelength * wavelength / const.c * flux_lamb
    return flux_nu.to(unit)


def flux_nu2lamb(
    flux_nu: Quantity,
    frequence: Quantity,
    unit: Any = DEFAULT_UNIT_FLUX_WAVELENGTH
) -> Quantity:
    flux_lamb: Quantity = frequence * frequence / const.c * flux_nu
    return flux_lamb.to(unit)
