#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 9 2022

@author: Patricia de Bruin
"""

import numpy as np
from scipy.special import gamma

def gammapdf(x, alpha, beta):
    return (np.power(beta, alpha) / gamma(alpha)) * (x ** (alpha - 1)) * np.exp(- beta * x)

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

alpha = 0.01

def gibbs(T, n, rho0, mu0, x, s, r, steps):
    """
    Parameters
    ----------
    T : number of iterations
    n : number of nurses
    rho0 : initial value parameter rho
    mu0 : initial value parameter mu
    x : vector incindents
    s : vector of shifts
    r : time-period
    steps: number of steps of 0.005 for range of rho (300 steps gives a maximum value of 1.5 for rho)

    Returns
    -------
    rho_hat : vector of all iterations of rho
    mu_hat : vector of all iterations of mu
    N_hat : vector of all iterations of probability 

    """
    rho_hat = [rho0]
    mu_hat = [mu0]
    N_hat = [0]

    for t in range(0, T):
        # lambda
        lamb_hat = []
        for i in range(0, n):
            lamb = np.random.gamma(shape = x[i] + rho_hat[t], scale = 1 / (s[i] + rho_hat[t] / mu_hat[t]), size = 1)
            lamb_hat.append(lamb)

        # mu (inverse-gamma)
        mu = float(np.random.gamma(shape =  n * rho_hat[t] - alpha, scale = 1 /(rho_hat[t] * sum(lamb_hat))))
        mu_hat.append(1 / mu)

        # rho (discretized)
        prob = []
        for j in range(1, steps + 1):
            p = (gammapdf(j * 0.005, 1, 1) / np.power(gamma(j * 0.005), n)) * np.exp(-(j * 0.005 / mu_hat[t + 1]) * sum(lamb_hat) + n * j * 0.005 * np.log(j * 0.005 / mu_hat[t + 1]) + (j * 0.005 - 1) * sum(np.log(lamb_hat)))
            prob.append(p)
    
        c = sum(prob)
        rho = float(np.random.choice(np.linspace(0.005, 0.005 * steps, steps), size = 1, replace = True, p = np.squeeze(prob / c)))
        rho_hat.append(rho)

        # P(N(t) >= 14)
        N = poissongamma(14, rho_hat[t + 1], r, mu_hat[t + 1])
        N_hat.append(N)
        
    return rho_hat, mu_hat, N_hat

def gibbs2(T, n, rho0, mu0, x, r, steps):
    """
    Parameters
    ----------
    T : number of iterations
    n : number of nurses
    rho0 : initial value parameter rho
    mu0 : initial value parameter mu
    x : vector incindents
    r : vector of shifts
    steps: number of steps of 0.005 for range of rho (300 steps gives a maximum value of 1.5 for rho)

    Returns
    -------
    rho_hat : vector of all iterations of rho
    mu_hat : vector of all iterations of mu
 
    """
    rho_hat = [rho0]
    mu_hat = [mu0]

    for t in range(0, T):
        # lambda
        lamb_hat = []
        for i in range(0, n):
            lamb = np.random.gamma(shape = x[i] + rho_hat[t], scale = 1 / (r[i] + rho_hat[t] / mu_hat[t]), size = 1)
            lamb_hat.append(lamb)

        # mu (inverse-gamma)
        mu = float(np.random.gamma(shape =  n * rho_hat[t] - alpha, scale = 1 /(rho_hat[t] * sum(lamb_hat))))
        mu_hat.append(1 / mu)

        # rho (discretized)
        prob = []
        for j in range(1, steps + 1):
            p = (gammapdf(j * 0.005, 1, 1) / np.power(gamma(j * 0.005), n)) * np.exp(-(j * 0.005 / mu_hat[t + 1]) * sum(lamb_hat) + n * j * 0.005 * np.log(j * 0.005 / mu_hat[t + 1]) + (j * 0.005 - 1) * sum(np.log(lamb_hat)))
            prob.append(p)
    
        c = sum(prob)
        rho = float(np.random.choice(np.linspace(0.005, 0.005 * steps, steps), size = 1, replace = True, p = np.squeeze(prob / c)))
        rho_hat.append(rho)
        
    return rho_hat, mu_hat