#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 2021

@author: Patricia de Bruin
"""

import numpy as np
import scipy.special
from scipy.special import gamma

from poissongamma import poissongamma

def systematic(i, K, K1, N):
    """
    Parameters
    ----------
    i : number of systematic fires
    K : total number of fires
    K1 : total number of accidental fires
    N : number of households in the Netherlands

    Returns
    -------
    upper bound for the probability q2 of having at least four fires in a household from which less than i have a systematic cause.

    """
    p = 0
    q = 0
    
    for j in range(5 - i, 4, 1):
        p += scipy.special.comb(K1, j, exact = True) * np.power((1 / N), j) * np.power((1 - 1 / N), (K1 - j)) * int((K - K1) / (4 - j)) / N
        
    for j in range(4, 21, 1):
        q += scipy.special.comb(K1, j, exact = True) * np.power((1 / N), j) * np.power((1 - 1 / N), (K1 - j))
    
    return p + q

"""
Created on Thu Nov 4 2021

@author: Patricia de Bruin
"""

def systematic2(i, K, K1, N, rho, mu, t):
    """
    Parameters
    ----------
    i : number of systematic fires
    K : total number of fires
    K1 : total number of accidental fires
    N : number of households in the Netherlands
    rho : parameter of the mixed Poisson model that models the heterogeneity
    mu : prameter of the mixed Poisson model that models the overall probability of an accidental fire per year
    t: time period (in years)

    Returns
    -------
    upper bound for the probability q2 of having at least four fires in a household from which less than i have a systematic cause.

    """
    p = 0
    
    a = 1/(1 + (t * mu) / rho)
    
    for j in range(5 - i, 4, 1):
        p += (gamma(j + rho) / (gamma(rho) * gamma(j + 1))) * np.power(a, rho) * np.power(1 - a, j) * int((K - K1) / (4 - j)) / N
       
    q = poissongamma(4, rho, t, mu)
        
    return p + q