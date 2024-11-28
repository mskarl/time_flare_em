import numpy as np
import utils
from collections import deque
import time


def get_mask(array, keyword, threshold):
    """ create the correct mask for the array
    
    Parameters:
    array: array or dataframe for which to create the mask
    keyword: the keyword to use for the array
    threshold: for which threshold to create the mask
    
    Returns:
    mask in the same shape as array
    """

    if keyword in ["SIGNAL", "ENERGY"]:
        mask = array[keyword] >= float(threshold)

    elif keyword == "AREA":
        mask = array[keyword] <= float(threshold)

    return mask


def run_em(sob_array, mask, tstart, tend, number_flares, n_start=0, n_trials=1000, start_sigma=500, start_ns=10, llh_conv=20, max_iter=500, seed=1):
    """ run expectation maximization and fit flares!
    
    Parameters:
    sob_array: [array] array with the signal over background values that should be used as weights for fitting the flares. Set all values to 1 for no weights.
    mask: [array] mask array (boolean) which values of sob_array should be included for the fit.
    tstart: [float] the starting time of the time period where events were recorded
    tend: [float] the end time of the time periods of recorded events
    number_flares: [int] how many Gaussian flares should be maximally allowd
    n_start: [int] in case one wants to split running the trials, set a starting value at which trial to start. Default is 0. 
    n_trials: [int] how many trials to run in this run. Default is 1000
    start_sigma: [float] the seed value for fitting the flare width. Set this to something broad. Default is 500 days. 
    start_ns: [float] the seed value for the strength. must be greater equal 0. start_ns * number_flares should not exceed total number of events
    llh_conv: [int] convergence criterium: number of iterations with no change in the likelihood to stop fit. Default = 20
    max_iter: [int] convergence criterium: number of maximal iterations of fit. Default = 500
    seed: [int] seed for random number generator for the time randomization. Default = 1
    
    Returns:
    List with [Log signal likelihood, log background likelihood, array with best-fit Gaussian means, array with best-fit Gaussian sigmas, array with best with number of events per Gaussian]
    """
    rng = np.random.default_rng(seed=seed)

    if n_start > len(sob_array):
        return []
    
    if n_start + n_trials > len(sob_array):
        n_trials = len(sob_array) - n_start

    llh_bg_array = []
    
    for trial in sob_array[n_start:n_start+n_trials]:
        times = rng.random(sum(mask)) * (tend - tstart) + tstart
        tmp_result = []

        # ensure that sum ns is smaller or equal N:
        if (start_ns * number_flares) > len(times):
            if len(times) <= number_flares: # set number of flares to be less than total events
                number_flares = len(times) - 1
            
            start_ns = max(len(times) / number_flares - 1, 0.5) # this should always be positive

        for my_weights in trial:
            weights = np.atleast_1d(my_weights)[mask]
            mean_T = np.linspace(tstart, tend, number_flares)
            sigma_T = np.array([start_sigma]*number_flares)
            ns_T = np.array([start_ns]*number_flares)


            llh1 = deque(range(llh_conv), llh_conv)
            i = 0
            t0 = time.time()
            while i < max_iter and (sum(np.abs(np.diff(llh1))) > 0):
                exp = utils.En(ns_T, mean_T, sigma_T, times, weights, tend-tstart, set_b_term=False)

                exp[0][exp[0] == 0] += 1e-10
                mean_T, sigma_T, ns_T = utils.Mn(exp[0], times, min_s=10)
                llh1.append(exp[1])
                i += 1
                mean_T, sigma_T, ns_T = np.array(mean_T), np.array(sigma_T), np.array(ns_T)

            tmp_result.append((exp[1], exp[2], mean_T, sigma_T, ns_T))

        llh_bg_array.append([tmp_result])

    return llh_bg_array


def start_run(sob, keyword, threshold, n_start, n_trials, start_ns=10):
    """ run the flare fit
    
    Parameters:
    sob: [dict] dictionary of signal over background (sob) values with thresholds as keys
    keyword: [string] quantity on which to apply the cut. Can be "AREA", "SIGNAL", or "ENERGY"
    threshold: [float] value for the threshold. This will enter the mask for the signal over background values
    n_start: [int] which trial to start with
    n_trials: [int] how many trials to fit this run
    start_ns: [float] starting seed of events belonging to a Gaussian flare
    
    Returns:
    list with results: [[Log signal likelihood, log background likelihood, array with best-fit Gaussian means, array with best-fit Gaussian sigmas, array with best with number of events per Gaussian]...]
    """

    mask = get_mask(alerts, keyword, threshold)
    tstart, tend = min(alerts["EVENTMJD"]), max(alerts["EVENTMJD"])
    sob_array = sob[str(threshold)]
    tmp_llh = run_em(sob_array, mask, tstart, tend, 20, n_start, n_trials, start_ns=start_ns)

    return tmp_llh

