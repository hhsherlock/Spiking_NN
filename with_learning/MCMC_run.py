#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 2

@author: yaning
"""

import numpy as np
import importlib
import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal, Uniform
from tqdm import tqdm

# functions and classes that i wrote
import run_network as run_network
import with_learning.learning_NN.Receptors as Receptors
import with_learning.learning_NN.Network as Network

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

def get_current_distance(currents):
    current_d = abs(np.nanmax(currents)-175)
    return current_d


# # try use MCMC result params
# path = "/home/yaning/Documents/Spiking_NN/without_learning/"
# samples = np.load(path + "MCMC_samples/static_std_initial_0.npy")
# cut_samples = samples[1000:, :]
# values = np.mean(cut_samples, axis=0)
# infer_names = Receptors.LigandGatedChannelFactory.infer_names
# infer_params = dict(zip(infer_names, values))

infer_names = Receptors.LigandGatedChannelFactory.infer_names
infer_params = Receptors.LigandGatedChannelFactory.infer_params


#-------------------------initialise MCMC---------------------------
samples = []
pure_samples = []

for i, key in enumerate(infer_params):
    if i <= 9:
        infer_params[key] = 0.5
    else:
        # factor is 20, multiply by 0.5 (initial)
        infer_params[key] = 10

initial_sample = []
for key in infer_params:
    initial_sample.append(infer_params[key])

# put first pure and normal samples in the record
pure_samples.append(np.zeros(len(infer_params)))
samples.append(initial_sample)

# run first round
# return voltages, currents, neuron_names
voltages, currents, neuron_names, firing_tstep = run_network.run(infer_params)

# evaluation scores
old_score = abs(235-len(firing_tstep))

#-----------------------officially run MCMC----------------------
num = 1000
factor = 20

for i in tqdm(range(num), desc="Processing", ncols=100):
    # # make the std decrease as MCMC keeps sampling
    # std = np.exp(-i*3/num)
    std = 5/(1+0.004*i)
    
    one_round_pure_sample = []
    one_round_sample = []
    
    temp_infer_params = infer_params
    
    last_pure_sample = pure_samples[-1]
    
    for j in range(len(infer_names)):
        # using j only for separate with and without factor
        temp_pure_sample = Normal(last_pure_sample[j], std).sample()
        one_round_pure_sample.append(temp_pure_sample)
        if j <= 9:
            temp_infer_params[infer_names[j]] = sigmoid(temp_pure_sample)
        else:
            temp_infer_params[infer_names[j]] = factor*sigmoid(temp_pure_sample)


    # run with sampled value
    voltages, currents, neuron_names, firing_tstep = run_network.run(temp_infer_params)
    
    # evaluation scores
    new_score = abs(235-len(firing_tstep))
    acceptance_ratio = old_score/new_score
    # print(old_score, new_score)
    
    # print(acceptance_ratio)

    u = np.random.uniform(0, 1)

    if acceptance_ratio >= u:
        # print("it is taken")
        old_score = new_score
        # current_d_old = current_d_new
        
        infer_params = temp_infer_params
        # if accept new, the add the pure for next round use
        pure_samples.append(one_round_pure_sample)

    
    for key in infer_params:
        one_round_sample.append(infer_params[key])
    
    samples.append(one_round_sample)
    
    

samples = np.array(samples)
pure_samples = np.array(pure_samples)


np.save(run_network.path + 'MCMC_samples/samples.npy', samples)
np.save(run_network.path + 'MCMC_samples/pure_samples.npy', pure_samples)