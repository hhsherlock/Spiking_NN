#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24

@author: yaning
"""

import HH as HH
import importlib
import numpy as np
import matplotlib.pyplot as plt

def run():
    # simulation environment
    deltaTms = 0.05
    Cm = 1
    # other Vm initialisation can lead to random firing
    Vm = 1.3458754117369027
    # 5000 cycles and every cycle is 0.01ms
    # in total 50ms
    pointCount = 5000
    voltages = np.empty(pointCount)
    times = np.arange(pointCount) * deltaTms #record the actual time
    stim = np.zeros(pointCount)
    stim[1500:4000] = 20  # create a square pulse

    # presynapse firing, for now has no meaning
    tsp_pre = [500,2000,3500]



