import pymc as pm
import pandas as pd
from patsy import dmatrix
import numpy as np
from bound_probs.utils.math import softplus_np, inverse_softplus_np
import pytensor.tensor as pt


def build_model(data, dof_splines=6):

    sub_ix, subject_mapping = pd.factorize(data.index.get_level_values('subject'))
    noise_ix, noise_mapping = pd.factorize(data['noiseType'])
    probrange_ix, probrange_mapping = pd.factorize(data['probRange'])

    # Different design matrix for low prob and high prob range!
    X_p = [dmatrix(f'bs(p, df={dof_splines}, degree=3, include_intercept=True, lower_bound=0.0, upper_bound=0.5) - 1', {'p':np.clip(data['p'], 0.0, 0.5)}),
        dmatrix(f'bs(p, df={dof_splines}, degree=3, include_intercept=True, lower_bound=.5, upper_bound=1.0) - 1', {'p':np.clip(data['p'], .5, 1.0)})]
    X_p = np.array(X_p)

    # Select appropriate design matrix per trial
    X_p = X_p[probrange_ix, np.arange(X_p.shape[1])]

    coords = {'subject': subject_mapping, 'spline_n': np.arange(dof_splines), 'noise_condition':df.noiseType.unique(), 'probRange':df.probRange.unique()}

    model = pm.Model(coords=coords)

    with model:

        beta_bias_mu = pm.Normal('beta_bias_mu', mu=0, sigma=.05, dims=('noise_condition', 'probRange', 'spline_n'))
        beta_bias_sd = pm.HalfCauchy('beta_bias_sd', .05, dims=('noise_condition', 'probRange', 'spline_n'))
        
        beta_sd_mu = pm.Normal('beta_sd_mu', mu=inverse_softplus_np(.1), sigma=1., dims=('noise_condition', 'probRange', 'spline_n'))
        beta_sd_sd = pm.HalfCauchy('beta_sd_sd', .05, dims=('noise_condition', 'probRange', 'spline_n'))

        subject_offset_bias = pm.Normal(f'beta_bias_offset', mu=0, sigma=1, dims=('subject', 'noise_condition', 'probRange', 'spline_n'))
        subject_offset_sd = pm.Normal(f'beta_d_offset', mu=0, sigma=1, dims=('subject', 'noise_condition', 'probRange', 'spline_n'))

        beta_bias = pm.Deterministic('beta_bias', beta_bias_mu + subject_offset_bias * beta_bias_sd, dims=('subject', 'noise_condition', 'probRange', 'spline_n'))
        beta_sd = pm.Deterministic('beta_sd', beta_sd_mu + subject_offset_sd * beta_sd_sd, dims=('subject', 'noise_condition', 'probRange', 'spline_n'))


        print(beta_bias[sub_ix, noise_ix, probrange_ix, :].shape.eval())

        print(X_p.shape)

        # pred_bias = pm.math.dot(X_p, beta_bias[sub_ix, noise_ix, probrange_ix, :])
        pred_bias = pt.sum(X_p * beta_bias[sub_ix, noise_ix, probrange_ix, :], axis=1)
        pred_sd = pt.softplus(pt.sum(X_p * beta_sd[sub_ix, noise_ix, probrange_ix, :], axis=1))

        ll = pm.Normal('pred', mu=pred_bias, sigma=pred_sd, observed=data['bias'].values)

    return model