import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid, fixed_quad
import scipy.stats as ss
from scipy.optimize import minimize

import probTools as tools

# Take input probability distribution and get encoded distribution of probability disstribution
# Does the internal tranformation and then adds internal noise as well.
def MI_efficient_encoding(p_input, sigma_rep):

    # Add sensory noise for each value of p_input
    p_m_given_p_input = tools.sensory_noise(tools.cdf_prob(p_input), sigma_rep=sigma_rep,
                                            grid=tools.rep_scale)

    return p_m_given_p_input

# subject's bayesian decoding mechanism for given representation m (done right now for all possible m's)
# theta_estimates gives where all rep_ori_grid points end up
def subject_prob_estimate(input_scale, sigma_rep, loss_exp=2):

    ## This one is used by subject for their bayesian decoded theta_estimate
    # Make a big array that for many thetas gives the probability of observing ms (subject likelihood)
    # input_scale x m_gen
    p_m_given_p_scale = tools.sensory_noise(tools.cdf_prob(input_scale), sigma_rep=sigma_rep,
                                    grid=tools.rep_scale)

    # input_scale x m (subject's bayesian decode)
    p_OutScale_given_m = p_m_given_p_scale * tools.prob_prior(input_scale, input_scale)[:, np.newaxis]
    p_OutScale_given_m = p_OutScale_given_m / trapezoid(p_OutScale_given_m, input_scale, axis=0)[np.newaxis, :]

    x0 = trapezoid(input_scale[:, np.newaxis]*p_OutScale_given_m, input_scale, axis=0)
    if loss_exp == 2:
        prob_estimates = x0
    else:
        #not done yet
        prob_estimates = x0

    return prob_estimates

# Given that a noisy encoding of stimulus was ensued and given that prob_estimates gives exact points where 
# each point in the bayesian observer's brain ends up. 
def output_prob_distribution(input_scale, p_input, sigma_rep, loss_exp = 2):

    p_m_given_p_input = MI_efficient_encoding(p_input, sigma_rep)
    prob_estimates = subject_prob_estimate(input_scale, sigma_rep, loss_exp=loss_exp)

    print(np.shape(prob_estimates))

    output_scale, output_prob_dist = tools.prob_transform(tools.rep_scale, prob_estimates, p_m_given_p_input)

    return output_scale, output_prob_dist 

def output_mean(input_scale, p_input, sigma_rep, loss_exp = 2):
    
    output_scale, output_prob_dist  = output_prob_distribution(input_scale, p_input, sigma_rep, loss_exp = loss_exp)
    mean_output = trapezoid(output_scale[np.newaxis, :]*output_prob_dist, output_scale, axis=1)
    
    return mean_output

def output_variance(input_scale, p_input, sigma_rep, loss_exp = 2):
    output_scale, output_prob_dist  = output_prob_distribution(input_scale, p_input, sigma_rep, loss_exp = loss_exp)
    mean_output = output_mean(input_scale, p_input, sigma_rep, loss_exp = 2)

    output_variances = np.sum((output_prob_dist * (output_scale[np.newaxis ,:] - mean_output[:, np.newaxis])**2), axis = 1)

    return output_variances