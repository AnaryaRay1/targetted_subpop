__author__ = "Anarya Ray; <anarya.ray@northwestern.edu>"

"""
Comp-1: SKN q, single-PL+single-G m1, Gaussian-chieff, Gaussian-chip
Comp-2: PL q, Broken-PL(+single-G?) m1, Gaussian-chieff, Gaussian-chip
Comp-3: G q, Broken-PLm1, Uniform-chieff, Beta-chip
"""
import gwpopulation
gwpopulation.set_backend("jax")
from gwpopulation.utils import xp, powerlaw, truncnorm, unnormalized_2d_gaussian, beta_dist
from gwpopulation.models.mass import two_component_single, double_power_law_primary_mass, double_power_law_peak_primary_mass
from lvk_mass_models import BrokenPowerLawPlusPeaksSmoothedMassDistribution, three_component_double_power_law_primary_mass, inspect
from gwpopulation.experimental.cosmo_models import CosmoModel
import jax.scipy.special as scs
from jax import jit

m1_grid = xp.logspace(xp.log10(3), xp.log10(300), 1000)
q_grid = xp.linspace(0.001, 1, 100)
m1s_grid, qs_grid = xp.meshgrid(m1_grid, q_grid)
z_max = 2.3
z_grid = xp.linspace(1e-6, z_max, 2500)
cosmo = CosmoModel("Planck15")
dVdz_grid = 4 * xp.pi * cosmo.cosmology(0.0).differential_comoving_volume(z_grid) #/(1.0+z_grid)

@jit
def powerlaw_one_plus_redshift(dataset, lamb):
    # norm = xp.trapz(dVdz_grid * (1.0 + z_grid)**(lamb - 1), z_grid)
    dVdz = xp.where(dataset["redshift"]<=z_max, xp.interp(dataset["redshift"], z_grid, dVdz_grid), 0.0)
    p_z = dVdz * ((1.0 + dataset["redshift"]) ** (lamb - 1.0))
    return p_z #/ norm
@jit
def differential_hypervolume(lamb):
    return xp.trapz(dVdz_grid * (1.0 + z_grid)**(lamb - 1), z_grid)


@jit
def smoothing(masses, mmin, mmax, delta_m):
        """
        Copied from: https://github.com/ColmTalbot/gwpopulation/blob/0071621071dedff6861df44b3b2a7c3698a51696/gwpopulation/models/mass.py#L660
        Owner: GWpopulation
        
        Apply a one sided window between mmin and mmin + delta_m to the
        mass pdf.

        The upper cut off is a step function,
        the lower cutoff is a logistic rise over delta_m solar masses.

        See T&T18 Eqs 7-8
        Note that there is a sign error in that paper.

        S = (f(m - mmin, delta_m) + 1)^{-1}
        f(m') = delta_m / m' + delta_m / (m' - delta_m)

        See also, https://en.wikipedia.org/wiki/Window_function#Planck-taper_window
        """
    
        shifted_mass = xp.nan_to_num((masses - mmin) / delta_m, nan=0)
        shifted_mass = xp.clip(shifted_mass, 1e-6, 1 - 1e-6)
        exponent = 1 / shifted_mass - 1 / (1 - shifted_mass)
        window = scs.expit(-exponent)
        window *= (masses >= mmin) * (masses <= mmax)
        return window
        

@jit
def trunc_eps_skewnorm(xx, mu, sigma, epsilon, high, low):
    """
    Author: Sharan Banagiri <sharan.banagiri@ligo.org>
    Taken from https://github.com/sharanbngr/gwpopulation/blob/8c7921eab41069614abbf34dc8840a94a2d93fa9/gwpopulation/utils.py#L173C1-L187C16
    """

    norm = 2**0.5 / xp.pi**0.5 / sigma
    norm /=  (1 - epsilon) * scs.erf((high - mu) / 2**0.5 / sigma / (1 - epsilon)) + \
         (1 + epsilon) * scs.erf((mu - low) / 2**0.5 / sigma / (1 + epsilon))


    prob_left = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma**2 * (1 + epsilon)**2))
    prob_right = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma**2 * (1 - epsilon)**2))

    prob = xp.where(xx > mu, prob_right, prob_left)

    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


@jit
def population_model_default(dataset, alpha_1, alpha_2, mlow_1, mmax, lam_0, lam_1, mpp_1, sigpp_1, mpp_2, sigpp_2, delta_m_1, beta, mlow_2, delta_m_2, mu_chi_eff, sigma_chi_eff, mu_chi_p, sigma_chi_p, lamb):
    p_m1_grid = three_component_double_power_law_primary_mass(m1_grid, alpha_1, alpha_2, mlow_1, mmax, lam_0, lam_1, mpp_1, sigpp_1, mpp_2, sigpp_2) * smoothing(m1_grid, mlow_1, mmax, delta_m_1)
    norm_m1 = xp.trapz(p_m1_grid, m1_grid)

    p_q_grid = powerlaw(qs_grid, beta, 1.0, mlow_2/m1s_grid) * smoothing(m1s_grid * qs_grid, mlow_2, m1s_grid, delta_m_2)
    norm_q = xp.nan_to_num(xp.trapz(p_q_grid, q_grid, axis=0)) * (delta_m_2 != 0) + 1 * (
            delta_m_2 == 0
        )

    p_m1 = three_component_double_power_law_primary_mass(dataset["mass_1"], alpha_1, alpha_2, mlow_1, mmax, lam_0, lam_1, mpp_1, sigpp_1, mpp_2, sigpp_2) * smoothing(dataset["mass_1"], mlow_1, mmax, delta_m_1) / norm_m1
    p_q = powerlaw(dataset["mass_ratio"], beta, 1.0, mlow_2/dataset["mass_1"]) * smoothing(dataset["mass_1"] * dataset["mass_ratio"], mlow_2, dataset["mass_1"], delta_m_2) / xp.interp(dataset["mass_1"], m1_grid, norm_q)
    p_chi = truncnorm(dataset["chi_eff"], mu=mu_chi_eff, sigma=sigma_chi_eff, high=1, low=-1) * truncnorm(dataset["chi_p"], mu=mu_chi_p, sigma=sigma_chi_p, high=1, low=0.0)
    p_z = powerlaw_one_plus_redshift(dataset, lamb)
    
    return xp.nan_to_num(p_m1 * p_q * p_chi * p_z, nan = 0.0)



@jit
def component_SKNq_PLPm1_chieffchip_g_PLz(dataset, alpha, mmin, mmax, lam, mpp, sigpp, delta_m,
                                        q_mu, q_sigma, q_epsilon, m2min,
                                        mu_chi_eff, mu_chi_p, sigma_chi_eff, sigma_chi_p, 
                                        lamb):

    p_m1_grid = two_component_single(m1_grid, alpha, mmin, mmax, lam, mpp, sigpp) * smoothing(m1_grid, mmin, mmax, delta_m)
    norm = xp.nan_to_num(xp.trapz(p_m1_grid, m1_grid))

    p_m1 = two_component_single(dataset["mass_1"], alpha, mmin, mmax, lam, mpp, sigpp) * smoothing(dataset["mass_1"], mmin, mmax, delta_m) / norm
    p_q = trunc_eps_skewnorm(dataset["mass_ratio"], q_mu, q_sigma, q_epsilon, 1.0, m2min/dataset["mass_1"])
    p_chieff_chip = truncnorm(dataset["chi_eff"], mu=mu_chi_eff, sigma=sigma_chi_eff, high=1, low=-1) * truncnorm(dataset["chi_p"], mu=mu_chi_p, sigma=sigma_chi_p, high=1, low=0.0)
    p_z = powerlaw_one_plus_redshift(dataset, lamb)
    
    return p_m1 * p_q * p_chieff_chip * p_z


@jit
def component_Nq_BPLm1_chieff_u_chip_g_PLz(dataset, alpha_1, alpha_2, mmin, mmax, break_mass, delta_m,
                                        q_mu, q_sigma, m2min, qmax,
                                        chi_eff_min, chi_eff_max, mu_chi_p, sigma_chi_p,
                                        lamb):

    break_fraction = (break_mass  - mmin) / (mmax - mmin)
    p_m1_grid = double_power_law_primary_mass(m1_grid, alpha_1, alpha_2, mmin, mmax, break_fraction) * smoothing(m1_grid, mmin, mmax, delta_m) 
    norm = xp.nan_to_num(xp.trapz(p_m1_grid, m1_grid))

    p_m1 = double_power_law_primary_mass(dataset["mass_1"], alpha_1, alpha_2, mmin, mmax, break_fraction) * smoothing(dataset["mass_1"], mmin, mmax, delta_m) / norm
    p_q = truncnorm(dataset["mass_ratio"], q_mu, q_sigma, qmax, m2min/dataset["mass_1"])
    p_chieff_chip = (dataset["chi_eff"]<=chi_eff_max) * (dataset["chi_eff"]>=chi_eff_min) * truncnorm(dataset["chi_p"], mu=mu_chi_p, sigma=sigma_chi_p, high=1, low=0.0) / (chi_eff_max - chi_eff_min)
    p_z = powerlaw_one_plus_redshift(dataset, lamb)
    
    return p_m1 * p_q * p_chieff_chip * p_z


@jit
def component_Nq_BPLm1_chieff_u_chip_sq_PLz(dataset, alpha_1, alpha_2, mmin, mmax, break_mass, delta_m,
                                        q_mu, q_sigma, m2min, qmax,
                                        chi_eff_min, chi_eff_max, at,
                                        lamb):

    break_fraction = (break_mass  - mmin) / (mmax - mmin)
    p_m1_grid = double_power_law_primary_mass(m1_grid, alpha_1, alpha_2, mmin, mmax, break_fraction) * smoothing(m1_grid, mmin, mmax, delta_m) 
    norm = xp.nan_to_num(xp.trapz(p_m1_grid, m1_grid))

    p_m1 = double_power_law_primary_mass(dataset["mass_1"], alpha_1, alpha_2, mmin, mmax, break_fraction) * smoothing(dataset["mass_1"], mmin, mmax, delta_m) / norm
    p_q = truncnorm(dataset["mass_ratio"], q_mu, q_sigma, qmax, m2min/dataset["mass_1"])
    p_chieff = (dataset["chi_eff"]<=chi_eff_max) * (dataset["chi_eff"]>=chi_eff_min)  / (chi_eff_max - chi_eff_min)
    # at = 1.5 * w
    p_chip = xp.where(dataset["chi_p"]<=at, dataset["chi_p"] / xp.sqrt((at**2 - dataset["chi_p"]**2)) /at , 0.0)
    p_z = powerlaw_one_plus_redshift(dataset, lamb)

    return p_m1 * p_q * p_chieff * p_chip * p_z


@jit
def component_Nq_BPLm1_chieff_g_chip_g_PLz(dataset, alpha_1, alpha_2, mmin, mmax, break_mass, delta_m,
                                        q_mu, q_sigma, m2min, qmax,
                                        mu_chi_eff, sigma_chi_eff, mu_chi_p, sigma_chi_p,
                                        lamb):

    break_fraction = (break_mass  - mmin) / (mmax - mmin)
    p_m1_grid = double_power_law_primary_mass(m1_grid, alpha_1, alpha_2, mmin, mmax, break_fraction) * smoothing(m1_grid, mmin, mmax, delta_m) 
    norm = xp.nan_to_num(xp.trapz(p_m1_grid, m1_grid))

    p_m1 = double_power_law_primary_mass(dataset["mass_1"], alpha_1, alpha_2, mmin, mmax, break_fraction) * smoothing(dataset["mass_1"], mmin, mmax, delta_m) / norm
    p_q = truncnorm(dataset["mass_ratio"], q_mu, q_sigma, qmax, m2min/dataset["mass_1"])
    p_chieff_chip =  truncnorm(dataset["chi_eff"], mu=mu_chi_eff, sigma=sigma_chi_eff, high=1, low=-1) * truncnorm(dataset["chi_p"], mu=mu_chi_p, sigma=sigma_chi_p, high=1, low=0.0) 
    p_z = powerlaw_one_plus_redshift(dataset, lamb)
    
    return p_m1 * p_q * p_chieff_chip * p_z
    
@jit
def component_PLq_BPLm1_chieffchip_g_PLz(dataset, alpha_1, alpha_2, mmin, mmax, break_mass, delta_m,
                                        q_beta, m2min,
                                        mu_chi_eff, mu_chi_p, sigma_chi_eff, sigma_chi_p,
                                        lamb):
    
    break_fraction = (break_mass  - mmin) / (mmax - mmin)
    p_m1_grid = double_power_law_primary_mass(m1_grid, alpha_1, alpha_2, mmin, mmax, break_fraction) * smoothing(m1_grid, mmin, mmax, delta_m) 
    norm_m1 = xp.nan_to_num(xp.trapz(p_m1_grid, m1_grid))
    p_q_grid = powerlaw(qs_grid, q_beta, 1.0, m2min/m1s_grid) * smoothing(m1s_grid * qs_grid, m2min, m1s_grid, delta_m)
    norm_q = xp.nan_to_num(xp.trapz(p_q_grid, q_grid, axis=0)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )
    
    p_m1 = double_power_law_primary_mass(dataset["mass_1"], alpha_1, alpha_2, mmin, mmax, break_fraction) * smoothing(dataset["mass_1"], mmin, mmax, delta_m) / norm_m1
    p_q = powerlaw(dataset["mass_ratio"], q_beta, 1.0, m2min/dataset["mass_1"]) * smoothing(dataset["mass_1"] *dataset["mass_ratio"], m2min, dataset["mass_1"], delta_m) / xp.interp(dataset["mass_1"], m1_grid, norm_q)
    p_chieff_chip = truncnorm(dataset["chi_eff"], mu=mu_chi_eff, sigma=sigma_chi_eff, high=1, low=-1) * truncnorm(dataset["chi_p"], mu=mu_chi_p, sigma=sigma_chi_p, high=1, low=0.0)
    p_z = powerlaw_one_plus_redshift(dataset, lamb)

    return p_m1 * p_q * p_chieff_chip * p_z

@jit
def component_PLq_BPLPm1_chieffchip_g_PLz(dataset, alpha_1, alpha_2, mmin, mmax, break_mass, delta_m, lam, mpp, sigpp,
                                        q_beta, m2min,
                                        mu_chi_eff, mu_chi_p, sigma_chi_eff, sigma_chi_p,
                                        lamb):
    
    break_fraction = (break_mass  - mmin) / (mmax - mmin)
    p_m1_grid = (lam * double_power_law_primary_mass(m1_grid, alpha_1, alpha_2, mmin, mmax, break_fraction)  + (1.0 - lam) * truncnorm(
        m1_grid, mu=mpp, sigma=sigpp, high=100.0, low=mmin
    )) * smoothing(m1_grid, mmin, mmax, delta_m)

    norm_m1 = xp.nan_to_num(xp.trapz(p_m1_grid, m1_grid))
    p_q_grid = powerlaw(qs_grid, q_beta, 1.0, m2min/m1s_grid) * smoothing(m1s_grid * qs_grid, m2min, m1s_grid, delta_m)
    norm_q = xp.nan_to_num(xp.trapz(p_q_grid, q_grid, axis=0)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )
    
    p_m1 = ( lam * double_power_law_primary_mass(dataset["mass_1"], alpha_1, alpha_2, mmin, mmax, break_fraction) + (1.0-lam) * truncnorm(
        dataset["mass_1"], mu=mpp, sigma=sigpp, high=100.0, low=mmin
    )) * smoothing(dataset["mass_1"], mmin, mmax, delta_m) / norm_m1
    p_q = powerlaw(dataset["mass_ratio"], q_beta, 1.0, m2min/dataset["mass_1"]) * smoothing(dataset["mass_1"] *dataset["mass_ratio"], m2min, dataset["mass_1"], delta_m) / xp.interp(dataset["mass_1"], m1_grid, norm_q)
    p_chieff_chip = truncnorm(dataset["chi_eff"], mu=mu_chi_eff, sigma=sigma_chi_eff, high=1, low=-1) * truncnorm(dataset["chi_p"], mu=mu_chi_p, sigma=sigma_chi_p, high=1, low=0.0)
    p_z = powerlaw_one_plus_redshift(dataset, lamb)

    return p_m1 * p_q * p_chieff_chip * p_z


@jit
def three_component_mixture_3_z(dataset,
                             alpha11, mmin1, mmax1, lam1, mpp1, sigpp1, delta_m, m2min, 
                             q_mu1, q_sigma1, q_epsilon1,
                             mu_chi_eff1, mu_chi_p1, sigma_chi_eff1, sigma_chi_p1,# eps_chi_eff1,
                             lamb_1,
                             alpha_21, alpha_22, mmin2, mmax2, break_mass2, # lam2, mpp2, sigpp2,
                             q_beta2, 
                             mu_chi_eff2, mu_chi_p2, sigma_chi_eff2, sigma_chi_p2,
                             lamb_2,
                             alpha31, alpha32, mmin3, mmax3, delta_m3, break_mass3, 
                             q_mu3, q_sigma3, qmax3, # qfrac3, q_beta3,
                             mu_chi_eff3, sigma_chi_eff3, mu_chi_p3, sigma_chi_p3, # , at_chip3 # chi_eff_min3, chi_eff_max3,
                             lamb_3,
                             # chi_eff_min3, chi_eff_max3,mu_chi_p3, sigma_chi_p3,
                             w_1, w_2):
    """
    Comp-1: SKN q, single-PL+single-G m1, Gaussian-chieff, Gaussian-chip
    Comp-2: PLq, Broken-PL m1, Gaussian-chieff, Gaussian-chip
    Comp-3: SKN q, single-PL m1, Uniform-chieff, Gaussian-chip
    """
    w_3 = 1.0 - w_1 - w_2
    vol = w_1 * differential_hypervolume(lamb_1) + w_2 * differential_hypervolume(lamb_2) + w_3 * differential_hypervolume(lamb_3)
    pdf = ( w_1 * component_SKNq_PLPm1_chieffchip_g_PLz(dataset, alpha11, mmin1, mmax1, lam1, mpp1, sigpp1, delta_m,
                                        q_mu1, q_sigma1, q_epsilon1, m2min,
                                        mu_chi_eff1, mu_chi_p1, sigma_chi_eff1, sigma_chi_p1,
                                        lamb_1) + 
            # w_1 * component_SKNq_PLPm1_chieff_g_chip_beta_PLz(dataset, alpha11, mmin1, mmax1, lam1, mpp1, sigpp1, delta_m,
            #                             q_mu1, q_sigma1, q_epsilon1, m2min,
            #                             mu_chi_eff1, sigma_chi_eff1, alpha_chip1, beta_chip1, at_chip1,
            #                             lamb_1) + 
            w_2 * component_PLq_BPLm1_chieffchip_g_PLz(dataset, alpha_21, alpha_22, mmin2, mmax2, break_mass2, delta_m,
                                        q_beta2, m2min,
                                        mu_chi_eff2, mu_chi_p2, sigma_chi_eff2, sigma_chi_p2,
                                        lamb_2) + 
            # w_2 * component_PLq_BPLPm1_chieffchip_g_PLz(dataset, alpha_21, alpha_22, mmin2, mmax2, break_mass2, delta_m, lam2, mpp2, sigpp2,
            #                             q_beta2, m2min,
            #                             mu_chi_eff2, mu_chi_p2, sigma_chi_eff2, sigma_chi_p2,
            #                             lamb_2) + 
            # w_2 * component_PLq_BPLm1_chieff_g_chip_beta_PLz(dataset, alpha_21, alpha_22, mmin2, mmax2, break_mass2, delta_m,
            #                             q_beta2, m2min,
            #                             mu_chi_eff2, sigma_chi_eff2, alpha_chip2, beta_chip2, at_chip2,
            #                             lamb_2) + 
            # w_3 * component_Nq_BPLm1_chieff_u_chip_g_PLz(dataset, alpha31, alpha32, mmin3, mmax3, break_mass3, delta_m3,
            #                             q_mu3, q_sigma3, m2min, qmax3,
            #                             chi_eff_min3, chi_eff_max3, mu_chi_p3, sigma_chi_p3,
            #                             lamb_3)
            w_3 * component_Nq_BPLm1_chieff_g_chip_g_PLz(dataset, alpha31, alpha32, mmin3, mmax3, break_mass3, delta_m3,
                                        q_mu3, q_sigma3, m2min, qmax3,
                                        mu_chi_eff3, sigma_chi_eff3, mu_chi_p3, sigma_chi_p3,
                                        lamb_3)
            # w_3 * component_Nq_BPLm1_chieff_u_chip_sq_PLz(dataset, alpha31, alpha32, mmin3, mmax3, break_mass3, delta_m3,
            #                             q_mu3, q_sigma3, m2min, qmax3,
            #                             chi_eff_min3, chi_eff_max3, at_chip3,
            #                             lamb_3)
            # w_3 * component_Nq_BPLm1_chieff_u_chip_beta_PLz(dataset, alpha31, alpha32, mmin3, mmax3, break_mass3, delta_m3,
            #                             q_mu3, q_sigma3, m2min, qmax3,
            #                             w_chi_eff3, alpha_chip3, beta_chip3, at_chip3,
            #                             lamb_3)
           
          ) / vol
    
    return xp.nan_to_num(pdf, nan = 0.0)

@jit
def three_component_mixture_3_restricted_z(dataset,
                             alpha11, mmin1, mmax1, lam1, mpp1, sigpp1, delta_m, m2min,
                             q_mu1, q_sigma1, q_epsilon1,
                             mu_chi_eff1, mu_chi_p1, sigma_chi_eff1, sigma_chi_p1,
                             lamb_1,
                             alpha_21, alpha_22, break_mass2,# lam2, sigpp2, mmax2,
                             q_beta2, 
                             mu_chi_eff2, mu_chi_p2, sigma_chi_eff2, sigma_chi_p2,
                             lamb_2,
                             alpha31, alpha32, break_mass3, mmin3, # delta_m3,
                             q_mu3, q_sigma3,
                             chi_eff_min3, chi, mu_chi_p3, sigma_chi_p3, #, at_chip3
                             lamb_3,
                             # mu_chi_p3, sigma_chi_p3,
                             w_1, w_23,
                             ):
    """
    Comp-1: SKN q, single-PL+single-G m1, Gaussian-chieff, Gaussian-chip
    Comp-2: PLq, Broken-PL m1, Gaussian-chieff, Gaussian-chip
    Comp-3: SKN q, single-PL m1, Uniform-chieff, Gaussian-chip
    """
    mmax2 = mmax1
    mmin2 = mmin1
    qmax3 = 1.0 
    mmax3 = 2.0 * mmax1
    delta_m3 = delta_m
    w_2 = (1.0 - w_1) * w_23
    # lamb_2 = lamb_1
    # alpha_chip3 =  alpha_chip3_1m + 1.0
    # beta_chip3 =  beta_chip3_1m + 1.0
    # chi_eff_min3 = -w_chi_eff3
    # chi_eff_max3 = w_chi_eff3
    mpp2 = break_mass2
    return three_component_mixture_3_z(dataset,
                             alpha11, mmin1, mmax1, lam1, mpp1, sigpp1, delta_m, m2min,
                             q_mu1, q_sigma1, q_epsilon1,
                             mu_chi_eff1, mu_chi_p1, sigma_chi_eff1, sigma_chi_p1,# eps_chi_eff1,
                             lamb_1,
                             alpha_21, alpha_22, mmin2, mmax2, break_mass2, # lam2, mpp2, sigpp2,
                             q_beta2, 
                             mu_chi_eff2, mu_chi_p2, sigma_chi_eff2, sigma_chi_p2,
                             lamb_2,
                             alpha31, alpha32, mmin3, mmax3, delta_m3, break_mass3, 
                             q_mu3, q_sigma3, qmax3, # qfrac3, q_beta3,
                             chi_eff_min3, chi_eff_max3, mu_chi_p3, sigma_chi_p3, #at_chip3
                             lamb_3,
                             w_1, w_2) #/ vol
                             # alpha11, mmin1, mmax1, lam1, mpp1, sigpp1, delta_m, m2min,
                             # q_mu1, q_sigma1, q_epsilon1,
                             # mu_chi_eff1, sigma_chi_eff1, alpha_chip1, beta_chip1, at_chip1,# eps_chi_eff1,
                             # lamb_1,
                             # alpha_21, alpha_22, mmin2, mmax2, break_mass2,
                             # q_beta2, 
                             # mu_chi_eff2, sigma_chi_eff2, alpha_chip2, beta_chip2, at_chip2,
                             # lamb_2,
                             # alpha31, alpha32, mmin3, mmax3, delta_m3, break_mass3, 
                             # q_mu3, q_sigma3, qmax3, # qfrac3, q_beta3,
                             # # w_chi_eff3, alpha_chip3, beta_chip3, at_chip3,
                             # chi_eff_min3, chi_eff_max3,mu_chi_p3, sigma_chi_p3,
                             # lamb_3,
                             # w_1, w_2)
@jit
def three_component_mixture_3_restricted(dataset,
                             alpha11, mmin1, mmax1, lam1, mpp1, sigpp1, delta_m, m2min,
                             q_mu1, q_sigma1, q_epsilon1,
                             mu_chi_eff1, mu_chi_p1, sigma_chi_eff1, sigma_chi_p1,
                             alpha_21, alpha_22, break_mass2,
                             q_beta2, 
                             mu_chi_eff2, mu_chi_p2, sigma_chi_eff2, sigma_chi_p2,
                             alpha31, alpha32, break_mass3, mmin3, # delta_m3,
                             q_mu3, q_sigma3,
                             w_chi_eff3, alpha_chip3_1m, beta_chip3_1m, at_chip3,
                             w_1, w_23,
                             lamb):
    """
    Comp-1: SKN q, single-PL+single-G m1, Gaussian-chieff, Gaussian-chip
    Comp-2: PLq, Broken-PL m1, Gaussian-chieff, Gaussian-chip
    Comp-3: SKN q, single-PL m1, Uniform-chieff, Gaussian-chip
    """
    mmax2 = mmax1
    mmin2 = mmin1
    qmax3 = 1.0 
    mmax3 = 2.0 * mmax1
    delta_m3 = delta_m
    # chi_eff_min3 = -w_chi_eff3
    # chi_eff_max3 = w_chi_eff3
    w_2 = (1.0 - w_1) * w_23
    lamb_2 = lamb_1 = lamb_3 = lamb
    alpha_chip3 =  alpha_chip3_1m + 1.0
    beta_chip3 =  beta_chip3_1m + 1.0
    return three_component_mixture_3_z(dataset,
                             alpha11, mmin1, mmax1, lam1, mpp1, sigpp1, delta_m, m2min,
                             q_mu1, q_sigma1, q_epsilon1,
                             mu_chi_eff1, mu_chi_p1, sigma_chi_eff1, sigma_chi_p1,
                             lamb_1,
                             alpha_21, alpha_22, mmin2, mmax2, break_mass2,
                             q_beta2, 
                             mu_chi_eff2, mu_chi_p2, sigma_chi_eff2, sigma_chi_p2,
                             lamb_2,
                             alpha31, alpha32, mmin3, mmax3, delta_m3, break_mass3, 
                             q_mu3, q_sigma3, qmax3,
                             w_chi_eff3, alpha_chip3, beta_chip3, at_chip3,
                             lamb_3,
                             w_1, w_2)
