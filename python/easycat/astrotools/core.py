from typing import Literal
from astropy import units as u
from astropy.units import Quantity
import numpy as np

FLUX_ZERO_DICT = {
    "W1": 309.540 * u.Jy,
    "W2": 171.787 * u.Jy,
    "J": 1594 * u.Jy,
    "H": 1024 * u.Jy,
    "K": 666.7 * u.Jy,
    "AB": 3631 * u.Jy
}

WAVELENGTH_EFFECT_DICT = {
    "W1": 3.4 * u.um,
    "W2": 4.6 * u.um,
    "J": 1.235 * u.um,
    "H": 1.662 * u.um,
    "K": 2.159 * u.um,
}

AstroBand = Literal[
    "u", "g", "r", "i", "z",
    "U", "B", "V",
    "W1", "W2", "W3", "W4",
    "J", "H", "K",
    "AB"
]

def get_flux_zero(band: AstroBand) -> Quantity:
    f0 = FLUX_ZERO_DICT.get(band, None)
    if f0 is None:
        raise Exception(f"Error band: {band}")
    
    return f0


def mag2flux(mag: float, band: AstroBand) -> Quantity:
    f0 = get_flux_zero(band)
    flux = f0 / (10**(0.4*mag))
    return flux.to(u.Jy)


def flux2mag(flux: Quantity, band: AstroBand) -> float:
    f0 = get_flux_zero(band)
    mag = 2.5 * np.log10(f0/flux)
    return mag.to_value()


def lamb2nu(wavelength: Quantity) -> Quantity:
    ...


def nu2lamb(frequence: Quantity) -> Quantity:
    ...


def flux_lamb2nu(flux_lamb: Quantity) -> Quantity:
    ...


def flux_nu2lamb(flux_nu: Quantity) -> Quantity:
    ...
