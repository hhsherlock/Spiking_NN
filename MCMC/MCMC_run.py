#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24

@author: yaning
"""

import numpy as np
import importlib
import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal, Uniform
from tqdm import tqdm

# functions and classes that i wrote
import run_network
import NN.Receptors as Receptors
import NN.Network as Network

importlib.reload(run_network)
importlib.reload(Receptors)
importlib.reload(Network)

np.set_printoptions(threshold=np.inf)

# functions
# get the minimum and maximum of the voltages
def get_min_max_distance(voltages):
    min_voltage = abs(np.nanmin(voltages) + 70)
    max_voltage = abs(np.nanmax(voltages) - 40.1)
    return min_voltage, max_voltage

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# original infer_params and names
infer_names = Receptors.LigandGatedChannelFactory.infer_names
infer_params = Receptors.LigandGatedChannelFactory.infer_params

# set all the initial parameters to 0 
# because the normal distribution's mean should at 0 
# so the initial value after sigmoid function will be 0.5 (in the middle)
for key in infer_params:
    infer_params[key] = 0

# run first round
voltages, firing, error_code, neurons = run_network.run(infer_params)

min_voltage_old, max_voltage_old = get_min_max_distance(voltages)


samples = []
pure_samples = []
num = 1500
for i in tqdm(range(num), desc="Processing", ncols=100):
    # sampling
    # std = np.exp(-i*3/num)
    std = 1
    factor = 30
    one_round_pure_sample = []
    
    temp_infer_params = infer_params
    
    for j in range(len(infer_names)):
        if j <= 9:
            temp_sample = Normal(infer_params[infer_names[j]], std).sample()
            one_round_pure_sample.append(temp_sample)
            temp_infer_params[infer_names[j]] = sigmoid(temp_sample)
        else:
            temp_sample = Normal(infer_params[infer_names[j]], std).sample()
            one_round_pure_sample.append(temp_sample)
            temp_infer_params[infer_names[j]] = factor*sigmoid(temp_sample)
    

    
    



    # run with sampled value
    voltages, firing, error_code, neurons = run_network.run(temp_infer_params)
    
    min_voltage_new, max_voltage_new = get_min_max_distance(voltages)
    
    # get the rates 
    min_voltage_rate = min_voltage_new/min_voltage_old
    max_voltage_rate = max_voltage_new/max_voltage_old
    
    # print(min_voltage_new, min_voltage_old, max_voltage_new, max_voltage_old)
    
    combine_evaluation = min_voltage_rate*max_voltage_rate
    # print(combine_evaluation)
    if combine_evaluation >= 1 or combine_evaluation > Uniform(0,1).sample():
        
        min_voltage_old = min_voltage_new
        max_voltage_old = max_voltage_new
        
        infer_params = temp_infer_params
    
    one_round_sample = []
    for name in infer_names:
        one_round_sample.append(infer_params[name])
    
    samples.append(one_round_sample)
    pure_samples.append(one_round_pure_sample)
    

samples = np.array(samples)
pure_samples = np.array(pure_samples)

np.save('static_std_initial_0.npy', samples)
np.save('static_std_initial_0_pure.npy', pure_samples)
