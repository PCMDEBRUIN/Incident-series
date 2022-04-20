#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 2021

@author: Patricia de Bruin
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

os.chdir(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Functies')))

from systematic import systematic

N = 7700000
K = 52000
i = 1 # Plug in the number of systematic fires

# data
K1 = np.linspace(0, K, K, dtype = int)
pdf = []

for k1 in range(0, K, 1):
    pdf.append(systematic(i, K, k1, N))

# plot
plt.plot(K1, pdf, color = 'maroon')
plt.title("Upper bound probability")
plt.xlabel("Number of accidental fires")
plt.ylabel("Probability less than 2 systematic fires")

p = (10 / N + (1.645 ** 2) / (2*N) - 1.645 * np.sqrt((10 / N) * (1 - (10 / N)) / N + (1.645 ** 2) /(4 * N ** 2))) / (1 + (1.645 ** 2) / N)
q2 = max(pdf)
q1 = p - q2
posteriorodds = q1/q2
posteriorprobability = posteriorodds/(1 + posteriorodds)

print('q1 is bounded from below by', q1)
print('q2 is bounded from above by', q2)
print('q1/q2 is bounded from below by', posteriorodds)
print('The posterior probability is bounded from below by', posteriorprobability)

# save figure in output folder
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))
time_str = time.strftime("%Y%m%d-%H%M%S")
# plt.savefig(os.path.join(output_dir, f'Systematic {time_str}.pdf'), format = 'pdf', dpi = 600)