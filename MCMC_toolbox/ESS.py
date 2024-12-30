import numpy as np

# copied from https://github.com/maarten-jung/master_thesis/blob/main/run_sampling_agent_distributions.py
def calc_ess(samples, step_size = 1, max_lag = 2000):
    # based on BEAST https://beast.community/
    # code adapted from the GitHub page (commit d2b36cf)
    # https://github.com/beast-dev/beast-mcmc/blob/f62cf395998ce1cd0538412177a13e672f96e9ac/src/dr/inference/trace/TraceCorrelation.java
    #
    # The effective samples size (ess) is defined as the number of samples devided by the (integrated) autocorrelation time (act).
    # The act is the delay at which the autocorrelation between the sample sequence and the delayed/shifted sample sequence is zero.
    # The act is calculated by first finding the approx. point where the autocovariance function (acf) is zero (indicated by the sum of 
    # adjacent values being less than zero). The acf is then assumed to be approx. linear and the act can thus be approximated by twice 
    # the area under the acf curve divided by the value of the acf at delay = 0.
    
    num_samples = len(samples)
    
    assert num_samples >= 10, "At least 10 samples needed!"
    
    if type(samples) is not np.ndarray:
        samples = np.array(samples)
    
    biggest_lag = min(num_samples-1, max_lag)
    
    # autocovariance function
    autocov = np.zeros(biggest_lag, dtype = np.float64)
    
    cent_samples = samples - samples.mean()
    
    for lag in range(biggest_lag):
      
        autocov[lag] = (cent_samples[:num_samples-lag] @ cent_samples[lag:]) / (num_samples - lag)
        
        if lag == 0: # (autocovariance at lag 0) == variance of samples
            integrated_autocov = autocov[lag]
        elif lag % 2 == 0:
            # sum of adjacent pairs of autocovariances must be positive (Geyer, 1992)
            sum_adj_pairs = autocov[lag-1] + autocov[lag]
            if sum_adj_pairs > 0:
                integrated_autocov += 2.0 * sum_adj_pairs
            else:
                break
        
    # integrated autocorrelation time
    if autocov[0] == 0:
        act = 0
    else:
        act = (step_size * integrated_autocov) / autocov[0]
    
    # effective sample size  
    if act == 0:
        ess = 1
    else:
        ess = (step_size * num_samples) / act
    
    return ess
