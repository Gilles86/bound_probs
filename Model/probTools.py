

import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid, fixed_quad
import scipy.stats as ss
from scipy.optimize import minimize

grid_points = 100
input_scale_full = np.linspace(0, 1, grid_points)
input_scale_half1 = np.linspace(0, 0.5, int(grid_points/2))
input_scale_half2 = np.linspace(0.5, 1.0, int(grid_points/2))
output_scale = input_scale_full

rep_scale = np.linspace(0.0, 1.0, grid_points)

# Prior here dictates the encoding of orientations and the cdf which governs the encoding transformation.
# Based on accuracy maximized codes that we propogate during training. 
# Changing this prior to something other than uniform will change the crossover point
def prob_prior(input_scale, x):
    x = np.asarray(x)
    prior = np.repeat(1, len(x)) / (len(x)*(input_scale[1] - input_scale[0]))
    return prior
    
def cdf_prob(x): # goes from 0 to 1
    x = np.asarray(x)
    cdf = x
    return cdf
    
def sensory_noise(m, sigma_rep, grid):
    # Ensure that m and sigma_rep are NumPy arrays
    m = np.atleast_1d(m)
    sigma_rep = np.atleast_1d(sigma_rep)

    if isinstance(sigma_rep, (int, float)):  # Check if sigma_rep is a scalar
        # If sigma_rep is a scalar, use it for all points in m
        sigma_rep = np.full_like(m, sigma_rep)

    # Ensure that grid is a NumPy array
    grid = np.array(grid)

    # Calculate the truncated normal distribution for each pair of mean and standard deviation
    truncBoth = ss.truncnorm.pdf(grid,
        (grid[..., 0] - m[:, np.newaxis]) / sigma_rep[:, np.newaxis],
        (grid[..., -1] - m[:, np.newaxis]) / sigma_rep[:, np.newaxis],
        m[:, np.newaxis],
        sigma_rep[:, np.newaxis]
    )
    return truncBoth



# Transform probability distribution over a grid of random variable to another random variable grid
# and then extrapolate outside the new grid with 0.
def prob_transform(grid, new_grid, p, bins=101, interpolation_kind='linear'):
    grid = np.array(grid)
    
    # For every bin in x_stim, calculate the probability mass within that bin
    dx = grid[..., 1:] - grid[..., :-1]
    p_mass = ((p[..., 1:] + p[..., :-1]) / 2) * dx

    if any(np.diff(new_grid)<=0): # For non monotonic transforms use histogram
        # Get the center of every bin
        x_value = new_grid[:-1] + dx / 2.
        ps = []
        for ix in range(len(p)):
            h, edges = np.histogram(x_value, bins=bins, weights=p_mass[ix], density=True)
            ps.append(h)

        ps = np.array(ps)
        new_grid = (edges[1:] + edges[:-1]) / 2

    else: #use the monotonic transformation formula analytic one
        ps = p
        ps[...,:] = ps[...,:]/abs(np.gradient(new_grid, grid))

    # ps_new = np.zeros(ps.shape)
    # We asssume here that the subject can never value any option outside of the val_estimates range
    # due to perceptual effects on the grid of values. 
    # new_grid_n = np.concatenate(([np.min(new_grid)-1e-6], new_grid, [np.max(new_grid)+1e-6]), axis=0)
    # ps_new = np.concatenate((np.zeros((len(ps), 1)), ps, np.zeros((len(ps), 1))), axis=1)

    f = interpolate.interp1d(new_grid, ps, axis=1,
                                 bounds_error=False, kind=interpolation_kind, fill_value=0.0)
    ps = f(grid)
    ps /= abs(trapezoid(ps, grid, axis=1)[:, np.newaxis])

    return grid, ps


def sensory_noise_2D(p, q, sigma_rep, grid):
    # Define the bounds for truncation in both dimensions
    p_lower_bound = (grid[0] - p) / sigma_rep
    p_upper_bound = (grid[-1] - p) / sigma_rep
    q_lower_bound = (grid[0] - q) / sigma_rep
    q_upper_bound = (grid[-1] - q) / sigma_rep
    
    # Calculate the joint truncated PDF for both dimensions
    p_pExt_q_qExt_given_m = ss.truncnorm.pdf(
        (grid - p) / sigma_rep, p_lower_bound, p_upper_bound) * \
        ss.truncnorm.pdf(
            (grid - q) / sigma_rep, q_lower_bound, q_upper_bound)
    
    return p_pExt_q_qExt_given_m

# Calculate how often distribution 1 is larger than distribution 2
# When both stimuli are gabors
def diff_dist(grid, p1, p2):
    p = []

    # p1 = p1[:, np.argsort(grid)]
    # p2 = p2[:, np.argsort(grid)]
    # grid = np.sort(grid)

    # grid: 1d
    # p1/p2: n_orienations x n(grid)
    cdf2 = integrate.cumtrapz(p2, grid, initial=0.0, axis=0)

    # for every grid point, distribution 1 is bigger than distribution 2
    # with a probability of being that value times the probability that dist
    # 2 is lower than that value
    prob = p1*cdf2

    p.append(prob)

    # Cummulative probability
    return integrate.trapz(p, grid)