import numpy as np
import celerite
from celerite import terms
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict
import scipy.stats as stats
from astropy.units import Quantity
from typing import Literal
import random
from random import choices

FRAC_PI_2 = np.pi/2
DOUBLE_PI = 2*np.pi
SQRT_2 = np.sqrt(2)

def beta_drw_sf(tau, sf_inf, tau_0, beta=1):
    return sf_inf * np.sqrt(
        1 - np.exp(-(np.abs(tau)/tau_0)**beta)
    )

def calc_sf(values, cadence):
    n = len(values)
    max_tau = (n - 1) * cadence
    tau = np.linspace(cadence, max_tau, num=n-1)
    sf = np.empty_like(tau)

    for i in range(n-1):
        k_cadence = i + 1
        i_left = np.arange(0, n-k_cadence)
        i_right = i_left + k_cadence

        diff = values[i_right] - values[i_left]
        sf[i] = np.sqrt(np.mean(diff*diff))

    return tau, sf

def calc_esf(sf_set):
    max_len = max(len(arr) for arr in sf_set)

    padded_arrays = np.array([
        np.pad(
            arr.astype(float), (0, max_len - len(arr)), 
            mode='constant', constant_values=np.nan
        ) for arr in sf_set
    ])

    mean_result = np.nanmean(padded_arrays, axis=0)

    return mean_result

def simulate_drw_lc(time, tau, sigma, mean):
    """
    Parameters
    ==========
    t: rest-frame time

    tau: DRW damping timescale

    sigma: rms of the drw light curve

    mean: mean of the light curve
    """
    log_a = 2*np.log(sigma)
    log_c = -np.log(tau)
    kernel = terms.RealTerm(log_a=log_a, log_c=log_c)
    
    gp = celerite.GP(kernel, mean=mean)
    gp.compute(time)
    
    y = gp.sample(size=1)[0]
    return y


def simulate_drw_esf(
        tau_set,
        sf_inf_set,
        n_lcurve: int=100,
        mean=1,
        cadence=5,
        dt=1,
        initial_length=9000,
        tmax=2000
    ):
    param_choice = random.choices(
        list(zip(tau_set, sf_inf_set)),
        k=n_lcurve
    )

    # simulate optical light curve
    sf_set = []
    for tau, sf_inf in param_choice:
        sigma = sf_inf/SQRT_2
        mjd_all = np.arange(0, tmax+initial_length+dt, dt)
        lc_all = simulate_drw_lc(mjd_all, tau=tau, sigma=sigma, mean=mean)
        lc_func = interp1d(mjd_all, lc_all)
        
        mjd = np.arange(min(mjd_all), max(mjd_all), cadence)
        value = lc_func(mjd)

        _, sf = calc_sf(value, cadence)
        sf_set.append(sf)
    
    esf = calc_esf(sf_set)
    delta_t = np.linspace(cadence, len(esf)*cadence, len(esf))
    return delta_t, esf


def get_flux_zero(band:Literal["W1", "W2", "g"]) -> Quantity:
    if band == "W1":
        # f0 = 306.681 * u.Jy
        f0 = 309.540 * u.Jy
    elif band == "W2":
        # f0 = 170.663 * u.Jy
        f0 = 171.787 * u.Jy
    elif band == "g":
        f0 = 3631 * u.Jy
    else:
        raise Exception(f"Error band: {band}")
    
    return f0

def flux2mag(flux:Quantity, band:Literal["W1", "W2", "g"]) -> float:
    f0 = get_flux_zero(band)
    mag = 2.5 * np.log10(f0/flux)
    return mag.to_value()

def mag2flux(mag, band:Literal["W1", "W2", "g"]):
    f0 = get_flux_zero(band)
    flux = f0 * 10**(-2/5*mag)
    return flux.to_value(u.Jy)

def generate_truncated_powerlaw(alpha, x_min, x_max, size=1):
    u = np.random.uniform(0, 1, size)
    
    if alpha == 1:
        samples = x_min * (x_max / x_min) ** u
    else:
        term1 = x_max ** (1 - alpha) - x_min ** (1 - alpha)
        term2 = u * term1 + x_min ** (1 - alpha)
        samples = term2 ** (1 / (1 - alpha))
    
    return samples

def generate_dusty_cloud(Rin, sigma, Y, p, ncloud):
    # theta = stats.truncnorm(-1, 1, loc=FRAC_PI_2, scale=sigma).rvs(size=ncloud)
    # phi = stats.uniform(0, DOUBLE_PI).rvs(size=ncloud)

    # Rout = Rin * Y
    # R = generate_truncated_powerlaw(p, Rin, Rout, size=ncloud)

    # return R, theta, phi

    # beta: vertical distribution
    sigma = np.rad2deg(sigma)
    theta_cos = np.random.uniform(0, np.cos((90-sigma)*np.pi/180), ncloud)

    theta = np.arccos(theta_cos)
    theta_degree = theta*180/np.pi

    beta_degree = 90 - theta_degree
    beta_degree = np.append(-beta_degree, beta_degree)

    # assuming gaussian distribution
    beta_prob = (1./(np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*(np.abs(beta_degree)/sigma)**2)

    beta_degree_random = choices(beta_degree, weights=beta_prob, k=len(beta_degree))
    beta = np.array(beta_degree_random) * np.pi / 180

    # phi
    phi = np.random.uniform(0, 2*np.pi, len(beta))

    # inner and outer radius
    Rout = Rin * Y
    RR = np.random.uniform(Rin, Rout, len(beta))
    R_prob = RR**(-p)
    R_random = choices(RR, weights=R_prob, k=len(beta))

    return R_random, beta, phi


class DustTorus:
    def __init__(self, Rin, sigma, Y, p, ncloud):
        self._Rin = Rin
        self._sigma = sigma
        self._Y = Y
        self._p = p
        self._ncloud = ncloud

        R, theta, phi = generate_dusty_cloud(Rin, sigma, Y, p, ncloud)
        self._clouds = {
            "R": R,
            "theta": theta,
            "phi": phi
        }
    
    def transfer_function(self, incl, *, dt=0.1):
        """ Copy from Li & Shen (2023)

        Parameters
        ==========
        incl: inclination angle in degree; 0 means face on and 90 means edge on

        dt: time interval of the transfer function, default 1 day

        Returns
        =======
        delay_t: tau in days

        delay_phi: transfer function
        """
        clouds = self._clouds
        R = clouds.get("R")
        theta = clouds.get("theta")
        phi = clouds.get("phi")
        Rin = self._Rin
        Y = self._Y
        
        cos_alpha = np.cos(theta)*np.cos(phi)*np.sin(incl) + np.sin(theta)*np.cos(incl)
        delay = (R*u.pc * (1-cos_alpha) / const.c).to(u.day).value

        # cos_alpha = np.sin(theta)*np.sin(incl)*np.cos(phi) + np.cos(incl)*np.cos(theta)
        # delay = (R*u.pc * (1-cos_alpha) / const.c).to_value(u.day)
        
        delay_hist = np.histogram(delay, bins=np.arange(0, 1e5, dt), density=True)
        delay_t = (delay_hist[1][:-1] + delay_hist[1][1:]) / 2.0
        delay_phi = delay_hist[0]

        tm = int(((2.0*Rin*Y)*u.pc / const.c).to(u.day).value)

        if tm < 10:
            tmax = 10
        else:
            idx = delay_t > tm
            zero_idx = list(delay_phi[idx]).index(0)
            tmax = int(1.1*delay_t[idx][zero_idx])

        delay_phi = delay_phi[delay_t <= tmax]
        delay_t = delay_t[delay_t <= tmax]

        return delay_t, delay_phi

    def simulate_dust_light_curve(
        self, tau_g, sigma_g, mean,
        incl, amp=1.0, dt=1.0,
        initial_length=9000, cadence=5, magnitude=False
    ):
        """
        Simulate idealized optical light curve from the DRW process. 
        Convolve the optical light curve with the torus transfer function to derive the MIR light curve.
        
        DRW parameters:
            tau_g: damping timescale
            sigma_g: rms variability
            flux_mean: mean flux
            
        Torus geomtry:
            incl: inclination angle in degree; 0 means face on and 90 means edge on
            dt: time interval of the transfer function (day); a small number is preferred (e.g., 1)
            amp: MIR/optical flux ratio
            
        Light curve:   
            cadence: cadence of the simulate light
            initial_length: initial length used to generate the MIR light curve; 
            
        Returns:
            simulated light curves in the optical and MIR.
        """
                
        # simulate transfer function
        delay_t, delay_phi = self.transfer_function(incl, dt=dt)
        tmax = max(delay_t)
        
        # simulate optical light curve
        mjd_all = np.arange(0, tmax+initial_length+dt, dt)
        gband_all = simulate_drw_lc(mjd_all, tau=tau_g, sigma=sigma_g, mean=mean)

        if magnitude:
            flux_g_all = mag2flux(gband_all, 'g')
        else:
            flux_g_all = gband_all


        # simulate MIR light curve
        flux_w_all = amp * np.convolve(delay_phi, flux_g_all) * dt
        idx1 = len(delay_t)
        idx2 = len(flux_g_all)
                
        flux_w_valid = flux_w_all[idx1:idx2]
        mjd_w_valid = mjd_all[idx1:idx2] 
        
        lc_w = interp1d(mjd_w_valid, flux_w_valid)
        lc_g = interp1d(mjd_all, flux_g_all)
        
        mjd_g = np.arange(min(mjd_all), max(mjd_all), cadence)
        mjd_w = np.arange(min(mjd_w_valid), max(mjd_w_valid), cadence)
        
        flux_w = lc_w(mjd_w)
        flux_g = lc_g(mjd_g)

        if magnitude:
            return flux2mag(flux_g*u.Jy, 'g'), mjd_g, flux2mag(flux_w*u.Jy, 'W1'), mjd_w
        else:
            return flux_g, mjd_g, flux_w, mjd_w
    
    def show_demo(
        self, tau_g, sigma_g, flux_mean,
        incl, amp=1.0, dt=1.0,
        initial_length=9000, cadence=5
    ):
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(nrows=2, ncols=2, figure=fig)

        ax_lc = fig.add_subplot(gs[0,:])
        ax_tf = fig.add_subplot(gs[1,0])
        ax_sf = fig.add_subplot(gs[1,1])

        flux_g, mjd_g, flux_w, mjd_w = self.simulate_dust_light_curve(
            tau_g=tau_g,
            sigma_g=sigma_g,
            flux_mean=flux_mean,
            incl=incl,
            cadence=cadence,
            amp=amp,
            dt=dt,
            initial_length=initial_length
        )

        ax_lc.plot(mjd_g, flux_g, linestyle="none", marker=".", markersize=1, color="tab:green")
        ax_lc.plot(mjd_w, flux_w, linewidth=2, color="tab:red")
        ax_lc.set_ylabel("Flux")
        ax_lc.set_xlabel("MJD - T0")

        t_delay, phi_delay = self.transfer_function(incl, dt=dt)
        ax_tf.plot(t_delay, phi_delay)

        sf_inf_g = np.sqrt(2)*sigma_g
        sf = calc_sf(flux_g, cadence)
        ax_sf.plot(sf[0], beta_drw_sf(sf[0], sf_inf_g, tau_g), color="k")
        ax_sf.plot(sf[0], sf[1], color="tab:green")

        sf = calc_sf(flux_w, cadence)
        ax_sf.plot(sf[0], sf[1], color="tab:red")

        ax_sf.loglog()
        return fig

def simulate_esf(
    dust_torus_param: Dict,
    # optical_param: Dict,
    tau_set,
    sf_inf_set,
    incl: float,
    mean_op: float = 1,
    n_lcurve: int=100,
    cadence=5,
    amp=1,
    magnitude=False,
):
    Rin = dust_torus_param.get("Rin", 0.5)
    sigma = dust_torus_param.get("sigma", np.pi/6)
    Y = dust_torus_param.get("Y", 1.5)
    p = dust_torus_param.get("p", 1)
    ncloud = dust_torus_param.get("ncloud", 20000)
    DT = DustTorus(Rin, sigma, Y, p, ncloud)

    # tau_op = optical_param.get("tau", 500)
    # sf_inf_op = optical_param.get("sf_inf", 0.2)
    # mean_op = optical_param.get("mean", 1.0)

    param_choice = random.choices(
        list(zip(tau_set, sf_inf_set)),
        k=n_lcurve
    )

    sf_g_set = []
    sf_w_set = []
    for i in range(n_lcurve):
        tau_op, sf_inf_op = param_choice[i]
        lc_gband, _, lc_wband, _ = DT.simulate_dust_light_curve(
            tau_g=tau_op,
            sigma_g=sf_inf_op/np.sqrt(2),
            mean=mean_op,
            incl=incl,
            cadence=cadence,
            amp=amp,
            magnitude=magnitude
        )

        # gmag = flux2mag(flux_g*u.Jy, "g")
        # wmag = flux2mag(flux_w*u.Jy, "W1")

        _, sf_g = calc_sf(lc_gband, cadence)
        _, sf_w = calc_sf(lc_wband, cadence)
        sf_g_set.append(sf_g)
        sf_w_set.append(sf_w)
    
    esf_g = calc_esf(sf_g_set)
    esf_w = calc_esf(sf_w_set)

    tau_g = np.linspace(cadence, len(esf_g)*cadence, len(esf_g))
    tau_w = np.linspace(cadence, len(esf_w)*cadence, len(esf_w))

    return tau_g, esf_g, tau_w, esf_w
