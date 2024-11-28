import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from astropy.stats.circstats import circmean
from astropy import units as u


def norm_pdf(norm_mu, norm_sigma, norm_x):
    """ A Gaussian pdf 
    
    Parameter:
    norm_mu: mean of Gaussian
    norm_sigma: sigma of Gaussian
    norm_x: values to evaluate
    
    Returns:
    pdf of Gaussian(mu, sigma)(x)
    """
    return 1 / np.sqrt(2 * np.pi * norm_sigma**2) * np.exp(-(norm_x - norm_mu)**2 / 2 / norm_sigma**2)


def En(ns, mu, sigma, t, sob, livetime, set_b_term=False):
    """
    Expectation step of expectation maximization
    ns: [list or array] the number of signal neutrinos, as weight for the gaussian flare
    mu: [list or array] the mean of the gaussian flare
    sigma: [list or array] sigma of gaussian flare
    t: [array] times of the events
    sob: [array] the signal over background values of events
    set_b_term: bool, whether to correct for a 10 degree window
    
    returns: probability of each point to belong to Gaussian (P(k|i)), log signal likelihood, log background likelihood
    """
    
    if set_b_term:
        b_term = (1 - np.cos(10 / 180 * np.pi)) / 2  # 2198.918456004788
    else:
        b_term = 1
        
    N = len(t)
    e_sig = []
    signal_norm = []
    for i in range(len(ns)):
        if ns[i] > 0:
            signal_norm.append(norm_pdf(mu[i], sigma[i], t))
            e_sig.append(signal_norm[-1] * sob * ns[i])
        else:

            signal_norm.append([0]*len(t))
            e_sig.append([0]*len(t))
    e_bg = (N - np.sum(ns)) / livetime / b_term 
    denom = sum(e_sig) + e_bg

    return [e / denom for e in e_sig], np.sum(np.log(1/N * (sum(np.array(e_sig)) + e_bg))), np.log(1 / livetime / b_term) * N


def Mn(e_sig, t, min_s=5):
    """
    maximization step of expectation maximization
    e_sig: [array] the weights for each event form the expectation step
    t: [array] the times of each event
    
    return: mu, sigma, ns (as lists) for each Gaussian
    """
    mu = []
    sigma = []
    ns = []
    for i in range(len(e_sig)):
        ns.append(np.sum(e_sig[i]))
        if ns[-1] == 0:
            mu.append(0)
            sigma.append(min_s)
        else:
            mu.append(np.average(t, weights=e_sig[i]))
            sigma.append(np.sqrt(np.average(np.square(t - mu[i]), weights=e_sig[i])))
        
    sigma = [max(min_s, s) for s in sigma]

    return mu, sigma, ns


def get_weighted_coords(x, sigma):
    """ get the weighted mean position and mean sigma
    x: [array] the position(s), usually RA and declination
    sigma: [array], (len(x), 2), error on positions
    """
    
    weighted_x_ = circmean(x*u.deg, weights = 1 / sigma**2, axis=0).value % 360
    weighted_sigma_ = (1 / np.sqrt(sum(1 / (sigma**2))))
    
    return weighted_x_, weighted_sigma_


def get_multiplet_weighted_coords(orig_index, multipl_indices, alerts):
    """ get the weighted coordinates for indices in alert dataframe
    orig_index: [int] the index of the alert event in the multiplet dictionary (the key)
    multipl_indices: [array like] the belonging indices of the multiplets of the alert with orig_index
    alerts: [pandas dataframe] dataframe with alert events
    """

    tmp_x = [(alerts.loc[orig_index].RA, alerts.loc[orig_index].DEC)]
    tmp_sigma = [[(alerts.loc[orig_index].RA_ERR_PLUS + alerts.loc[orig_index].RA_ERR_MINUS) / 2, 
                (alerts.loc[orig_index].DEC_ERR_PLUS + alerts.loc[orig_index].DEC_ERR_MINUS) / 2]]
    
    for tmp_index_2 in multipl_indices:
        tmp_x.append((alerts.loc[tmp_index_2].RA, alerts.loc[tmp_index_2].DEC))
        tmp_sigma.append([(alerts.loc[tmp_index_2].RA_ERR_PLUS + alerts.loc[tmp_index_2].RA_ERR_MINUS) / 2, 
                (alerts.loc[tmp_index_2].DEC_ERR_PLUS + alerts.loc[tmp_index_2].DEC_ERR_MINUS) / 2])

    tmp_x, tmp_sigma = np.atleast_1d(tmp_x), np.atleast_1d(tmp_sigma)
    tmp_coords = get_weighted_coords(tmp_x, tmp_sigma)

    return tmp_coords
    

def go_through_multiplet_dict(multiplet_dict, alerts):
    """ go through the multiplet dictionary and get the weighted positions and means for the multiplets
    multiplet_dict: [dict] with the first index as key and the mutliplet indices as entries
    alerts: [pandas dataframe] dataframe with alert events
    """
    
    weighted_x_, weighted_sigma_ = {}, {}

    for tmp_index in multiplet_dict:
        
        tmp_mult_indices = multiplet_dict[tmp_index]

        if len(tmp_mult_indices) > 0:

            weighted_x_[tmp_index], weighted_sigma_[tmp_index] = get_multiplet_weighted_coords(tmp_index, 
                                                                                               tmp_mult_indices, 
                                                                                               alerts)
       
        else:
            continue
            
            
    return weighted_x_, weighted_sigma_


def go_through_threshold_multiplet_dict(threshold_multiplet_dict, alerts):
    """ go throught the dictionary with tresholds and the respective multiplet indices and get the weighted positions and sigmas
    area_multiplet_dict: [dict] with thresholds as keys
    alerts: [pandas dataframe] dataframe with alert events 
    """
    
    weighted_x_, weighted_sigma_ = {}, {}
    
    for tmp_area in threshold_multiplet_dict:
        tmp_multiplets = threshold_multiplet_dict[tmp_area][1]
        
        threshold_w_x, threshold_w_sigma = go_through_multiplet_dict(tmp_multiplets, alerts)
        
        if len(threshold_w_x) > 0:
            weighted_x_[tmp_threshold] = threshold_w_x
            weighted_sigma_[tmp_threshold] = threshold_w_sigma
    
    return weighted_x_, weighted_sigma_
    
    
def angular_distance(ra1, dec1, ra2, dec2, psi_floor=None):
    """Calculates the angular separation on the shpere between two vectors on
    the sphere. Formula from Wikipedia...

    Parameters
    ----------
    ra1 : float | array of float
        The right-ascention or longitude coordinate of the first vector in
        radians.
    dec1 : float | array of float
        The declination or latitude coordinate of the first vector in radians.
    ra2 : float | array of float
        The right-ascention or longitude coordinate of the second vector in
        radians.
    dec2 : float | array of float
        The declination coordinate of the second vector in radians.
    psi_floor : float | None
        If not ``None``, specifies the floor value of psi.

    Returns
    -------
    psi : float | array of float
        The calculated angular separation value(s).
    """
    delta_ra = np.abs(ra1 - ra2)
    delta_dec = np.abs(dec1 - dec2)

    x = np.sin(delta_dec / 2.)**2. +\
        np.cos(dec1) * np.cos(dec2) * np.sin(delta_ra / 2.)**2.

    # Handle possible floating precision errors.
    x[x < 0.] = 0.
    x[x > 1.] = 1.

    psi = np.arccos(np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))

    if psi_floor is not None:
        psi = np.where(psi < psi_floor, psi_floor, psi)

    return psi

