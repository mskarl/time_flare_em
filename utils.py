import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#from scipy.stats import norm

from astropy.stats.circstats import circmean
from astropy import units as u



def is_in_ellipse(points, ra, dec, ra_err, dec_err):
    """ checks if points are in an ellipse centered at ra, dec with axes defined by ra_err and dec_err
    
    Parameter:
    points: [array] points for which to check if they are in the ellipse
    ra: [float] right ascencsion value of the ellipse center
    dec: [float] declination value of the ellipse center
    ra_err: (negative right ascencion error, positive right ascencion error) tuple with the uncertainty values of the right ascension (+- error). Note that the negative error must be negative (with a minus). 
    dec_err: (negative declination error, positive right declination) tuple with the uncertainty values of the declination (+- error). Note that the negative error must be negative (with a minus). 
    
    Returns:
    mask for events (points) within error ellipse
    """
    ra_min, ra_max = min(ra_err), max(ra_err)
    dec_min, dec_max = min(dec_err), max(dec_err)
    
    # get relative coordinates, check for cases close to 0 or 360 degrees
    tmp_ra = points["RA"] - ra 

    if (ra + ra_max) >= 360:
        tmp_ra = tmp_ra % 360
        
    if (ra + ra_min) <= 0:
        tmp_ra = -tmp_ra % 360
        tmp_ra = -tmp_ra
    
    tmp_dec = points["DEC"] - dec 

    # select 1st quadrant
    mask1 = ((tmp_ra >= 0) & (tmp_ra <= ra_max))
    mask1 = mask1 & (((tmp_dec >= 0) & (tmp_dec <= dec_max)))
    # select within contour
    mask1 = mask1 & (((tmp_ra**2)  / (ra_max**2)) + ((tmp_dec**2) / (dec_max**2)) <= 1)

    # select 2nd quadrant
    mask2 = ((tmp_ra >= 0) & (tmp_ra <= ra_max))
    mask2 = mask2 & (((tmp_dec <= 0) & (tmp_dec >= dec_min)))
    # select within contour
    mask2 = mask2 & (((tmp_ra**2)  / (ra_max**2)) + ((tmp_dec**2) / (dec_min**2)) <= 1)

    # select 3rd quadrant
    mask3 = ((tmp_ra <= 0) & (tmp_ra >= ra_min))
    mask3 = mask3 & (((tmp_dec <= 0) & (tmp_dec >= dec_min)))
    # select within contour
    mask3 = mask3 & (((tmp_ra**2)  / (ra_min**2)) + ((tmp_dec**2) / (dec_min**2)) <= 1)

    # select 4th quadrant
    mask4 = ((tmp_ra <= 0) & (tmp_ra >= ra_min))
    mask4 = mask4 & (((tmp_dec >= 0) & (tmp_dec <= dec_max)))
    # select within contour
    mask4 = mask4 & (((tmp_ra**2)  / (ra_min**2)) + ((tmp_dec**2) / (dec_max**2)) <= 1)

    mask = mask1 | mask2 | mask3 | mask4

    return mask


# def arc_patch(center, radius1, radius2, theta1, theta2, ax=None, resolution=15, closed=True, **kwargs):
#     # make sure ax is not empty
#     if ax is None:
#         ax = plt.gca()
        
#     # generate the points
#     theta = np.linspace(theta1, theta2, resolution)
#     points = np.vstack((np.append(radius1*np.cos(np.deg2rad(theta)) + center[0], center[0]), 
#                         np.append(radius2*np.sin(np.deg2rad(theta)) + center[1], center[1])))

#     # build the polygon and add it to the axes
#     poly = Polygon(points.T, closed=closed, **kwargs)
#     ax.add_patch(poly)
        
#     return poly


def get_multiplet_index_dictionary(alerts):
    """Go trough the provided alert and check for overlapping/touching events
    
    Parameters:
    alerts: [pandas dataframe] dataframe containing events with keys ["RA"], ["DEC"], ["RA_ERR_PLUS"], ["RA_ERR_MINUS"], 
                                            ["DEC_ERR_PLUS"], ["DEC_ERR_MINUS"]
    
    Returns:
    Dictionary with alert index as key and a list of touching alert indices as entries. 
    
    """
    
    multiplets = {}

    # go through all alert events and check which events are touching
    for tmp_index in alerts.index:
        tmp_alerts = alerts.loc[tmp_index]
        # check how many alerts lie within error region
        # go through areas of alerts and check
        alerts_in_ellipse = None
        
        # go through different quadrants of uncertainty regions. theta is the angle of the ellipse
        for theta, err_ra, err_dec in zip([0, 90, 180, 270], 
                                          ["RA_ERR_PLUS", "RA_ERR_MINUS", "RA_ERR_MINUS", "RA_ERR_PLUS"], 
                                          ["DEC_ERR_PLUS", "DEC_ERR_PLUS", "DEC_ERR_MINUS", "DEC_ERR_MINUS"]):

            t = np.linspace(theta, theta + 90, 20)
            # create points along the edge of the ellipse for the respective quadrant. This is done for all alerts
            ra_ell = alerts[err_ra].values.reshape((len(alerts), 1)) * np.cos(np.deg2rad(t)) +  alerts["RA"].values.reshape((len(alerts),1)) 
            dec_ell = alerts[err_dec].values.reshape((len(alerts), 1)) * np.sin(np.deg2rad(t)) +  alerts["DEC"].values.reshape((len(alerts),1))
            
            # array with all edge values of all alert uncertainty ellipses for the respective quadrant
            points = np.lib.recfunctions.merge_arrays((ra_ell, dec_ell)).reshape((len(alerts), 20))

            points.dtype = [("RA", "f8"), ("DEC", "f8")]
            
            # check for the points inside the ellipse of tmp_alert
            in_ellipse_mask = is_in_ellipse(points, tmp_alerts["RA"], tmp_alerts["DEC"], 
                                            (tmp_alerts["RA_ERR_PLUS"], -tmp_alerts["RA_ERR_MINUS"]), 
                                            (tmp_alerts["DEC_ERR_PLUS"], -tmp_alerts["DEC_ERR_MINUS"]))

            if np.any(in_ellipse_mask):
                # get indices of which alerts touch tmp_alerts uncertainty ellipse
                if alerts_in_ellipse is None:
                    alerts_in_ellipse = alerts[in_ellipse_mask].index.drop_duplicates()
                else:
                    alerts_in_ellipse = alerts_in_ellipse.append(alerts[in_ellipse_mask].index).drop_duplicates()

                if tmp_index in alerts_in_ellipse:
                    alerts_in_ellipse = alerts_in_ellipse.drop(tmp_index)
        # create a dictionary with key of tmp_alert index and each touching alert index as entry
        multiplets[tmp_index] = alerts_in_ellipse


    # check "double" multiplets. Keep either the one with the larger number of coincidences or keep the 
    # first association 

    # go through key, check if there are multiplets
    remove_multiplets_for_indices = []
    for _index in multiplets:
        if len(multiplets[_index]) > 0:
            # check if the _index is in some of the multiplets entries as well
            for _mult_index in multiplets[_index]:
                if _index in multiplets[_mult_index] :
                    # compare lengths and keep the one with more associations 
                    #(or if they are equal, just keep the first)
                    if len(multiplets[_index]) < len(multiplets[_mult_index]):
                        remove_multiplets_for_indices.append(_index)
                        
                    elif len(multiplets[_index]) > len(multiplets[_mult_index]):
                        remove_multiplets_for_indices.append(_mult_index)
                        
                    else:
                        remove_multiplets_for_indices.append(max(_index, _mult_index))
                        
    for _index in remove_multiplets_for_indices:
        multiplets[_index] = []
    return multiplets



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


def go_through_area_multiplet_dict(area_multiplet_dict, alerts):
    """ go throught the dictionary with areas and the respective multiplet indices and get the weighted positions and sigmas
    area_multiplet_dict: [dict] with areas as keys
    alerts: [pandas dataframe] dataframe with alert events 
    """
    
    weighted_x_, weighted_sigma_ = {}, {}
    
    for tmp_area in area_multiplet_dict:
        tmp_multiplets = area_multiplet_dict[tmp_area][1]
        
        area_w_x, area_w_sigma = go_through_multiplet_dict(tmp_multiplets, alerts)
        
        if len(area_w_x) > 0:
            weighted_x_[tmp_area] = area_w_x
            weighted_sigma_[tmp_area] = area_w_sigma
    
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

