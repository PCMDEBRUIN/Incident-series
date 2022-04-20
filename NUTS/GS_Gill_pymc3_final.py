#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 2022

@author: Patricia de Bruin
"""

import os
import time
import pyreadr

import arviz as az
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

from scipy.special import gamma

az.style.use(['default'])

# import data simulated data set from R
os.chdir('/Users/patricia/Documents/Mathematical Sciences/Research Thesis/Python/Plot')
# dataset = pyreadr.read_r('/Users/patricia/Documents/Mathematical Sciences/Research Thesis/R/Plot/Gibbs sampler/Gibbs_sampler_dataset.Rdata')

# x = dataset['x'].to_numpy() # number of incidents
# x_data = np.squeeze(x)
# x_data[0] = 50 # outlier

n = 27 * 3 # number of nurses
r = 201 # number of shifts

# parameter choice of Gill et al.
rho0 = 1
mu0 = 27/1734

x_data = np.random.negative_binomial(n = rho0, p = 1 / (1 + r * mu0 / rho0), size = n) # vector of the number of incidents

r_data = np.empty(n)
r_data.fill(r) # vector of the number of shifts

niter = 1000
chains = 4

with pm.Model() as mixed_poisson:
    # define priors
    rho = pm.Gamma('rho', alpha = 1, beta = 1)
    mu = pm.Beta('mu', alpha = 0.01, beta = 1)
    
    # model heterogeneity
    lamb = pm.Gamma('lamb', alpha = rho, beta = rho / mu, shape = n)
    
    # observations
    x_obs = pm.Poisson('x_obs', mu = r * lamb, observed = x_data)
    
    # model specifications
    step = pm.NUTS()

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))
time_str = time.strftime("%Y%m%d-%H%M%S")

with mixed_poisson:
    # draw posterior samples
    trace = pm.sample(niter, step = step, return_inferencedata = False, chains = chains, cores = 1)
    
    # specifications plots
    var_names = ['rho', 'mu']
    chain_prop = {'color': ['teal', 'indianred', 'lightseagreen', 'maroon'], 'alpha': [0.5]}  
    lines = (('rho', {}, np.mean(trace['rho'])), ('mu', {}, np.mean(trace['mu'])))
    
    # traceplot of the parameters
    pm.plot_trace(trace, figsize = (12,8), var_names = ['rho', 'mu'], lines = lines, chain_prop = chain_prop, compact = True, legend = True)
    plt.savefig(os.path.join(output_dir, f'Traceplot rho and mu {time_str}.pdf'), format = 'pdf', dpi = 600)

    # forest plots of the parameters
    pm.plot_forest(trace, var_names = 'rho', colors = 'maroon')
    plt.savefig(os.path.join(output_dir, f'Forestplot rho {time_str}.pdf'), format = 'pdf', dpi = 600)

    pm.plot_forest(trace, var_names = 'mu', colors = 'maroon')
    plt.savefig(os.path.join(output_dir, f'Forestplot mu {time_str}.pdf'), format = 'pdf', dpi = 600)
    plt.show()
    
# summary of parameters
with mixed_poisson:
    summary = pm.summary(trace, round_to = 10, kind = 'all')
    print(summary)


# function for p-value of mixed poisson model
def poissongamma(n, rho, t, mu):
    p = 1 / (1 + t * mu / rho)
    q = 0
    
    for k in range(0, n):
        q += gamma(k + rho) / (gamma(rho) * gamma(k + 1)) * np.power(p, rho) * np.power(1 - p, k)
    
    return 1 - q

# traceplot P(N(t) >= 14)
x = np.linspace(1, niter, niter, dtype = 'int')
N = poissongamma(14, rho = trace['rho'], t = r, mu = trace['mu'])
N0 = poissongamma(14, rho = rho0, t = r, mu = mu0) # true value

plt.plot(x, N[0:niter], label = '0', color = 'teal', alpha = 0.5)
plt.plot(x, N[niter:2 * niter], label = '1', color = 'indianred', alpha = 0.5)
plt.axhline(y = np.mean(N), color = 'black', alpha = 0.4)
plt.plot(x, N[2 * niter:3 * niter], label = '2', color = 'lightseagreen', alpha = 0.5)
plt.plot(x, N[3 * niter:4 * niter], label = '3', color = 'maroon', alpha = 0.5)
plt.xlim(0, niter)
plt.legend(loc = 'upper right', title = 'chain')
plt.title('$P(N(t) \geq 14)$')
plt.savefig(os.path.join(output_dir, f'Traceplot N {time_str}.pdf'), format = 'pdf', dpi = 600)