#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 2021

@author: Patricia de Bruin
"""

from scipy.special import gamma
import numpy as np

def poissongamma(n, rho, t, mu):
    """
    Parameters
    ----------
    n : value of p-value P(N(t) >= n)
    rho : parameter of the mixed Poisson model that models the heterogeneity
    t : time period (given in desired time unit)
    mu : parameter of the mixed Poisson model that models the overall probability of having an incident per time unit

    Returns
    -------
    p-value P(N(t) >= n)

    """
    p = 1/(1 + (t * mu) / rho)
    q = 0
    
    for k in range(0, n):
        q += gamma(k + rho) / (gamma(rho) * gamma(k + 1)) * np.power(p, rho) * np.power(1 - p, k)
    
    return 1 - q