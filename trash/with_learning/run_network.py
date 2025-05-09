#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 2

@author: yaning
"""

import importlib
import numpy as np
import matplotlib.pyplot as plt

# my own class files
import with_learning.learning_NN.Receptors as Receptors
import with_learning.learning_NN.Network as Network


importlib.reload(Receptors)
importlib.reload(Network)



pointCount = 5000
deltaTms = 0.05
times = np.arange(pointCount) * deltaTms
initial_Vm = 1.3458754117369027
# Neuron: deltaTms, I, Vm, fire times

# get the input neuron firing patterns
path = "/home/yaning/Documents/Spiking_NN/with_learning/"
input_pattern = np.load(path + "dataset.npy") 
output_pattern = np.load(path + "output.npy")

input_pattern = input_pattern[1]


def run(infer_params):
    Receptors.LigandGatedChannelFactory.infer_params = infer_params
    
    # Neuron: deltaTms, I, Vm, Name
    neuron_input_0 = Network.Neuron(deltaTms, 0, initial_Vm, "input_0")
    neuron_input_1 = Network.Neuron(deltaTms, 0, initial_Vm, "input_1")
    neuron_input_2 = Network.Neuron(deltaTms, 0, initial_Vm, "input_2")

    neuron_excite_main = Network.Neuron(deltaTms, 0, initial_Vm, "excite_main")
    neuron_excite_sub = Network.Neuron(deltaTms, 0, initial_Vm, "excite_sub")

    neuron_inhibit_main = Network.Neuron(deltaTms, 0, initial_Vm, "inhibit_main")
    neuron_inhibit_sub = Network.Neuron(deltaTms, 0, initial_Vm, "inhibit_sub")

    neuron_output = Network.Neuron(deltaTms, 0, initial_Vm, "output")

    neurons = [neuron_input_0, neuron_input_1, neuron_input_2, neuron_excite_main, neuron_excite_sub, 
            neuron_inhibit_main, neuron_inhibit_sub, neuron_output]

    neuron_names = ["input_0", "input_1", "input_2",
                    "excite_main", "excite_sub", "inhibit_main", "inhibit_sub", "output"]





    # create synapse/connection (send neuron, receive neuron)
    control = Network.Control(deltaTms, initial_Vm)




    #*********************full layer***************************
    # ----------------first input layer------------------------
    control.create_synapse(neuron_input_0, neuron_excite_main, "AMPA")
    control.create_synapse(neuron_input_1, neuron_excite_main, "AMPA")
    control.create_synapse(neuron_input_2, neuron_excite_main, "AMPA")

    control.create_synapse(neuron_input_0, neuron_inhibit_main, "GABA")
    control.create_synapse(neuron_input_1, neuron_inhibit_main, "GABA")
    control.create_synapse(neuron_input_2, neuron_inhibit_main, "GABA")



    # ----------------self recurrent layer----------------
    control.create_synapse(neuron_excite_main, neuron_excite_sub, "AMPA+NMDA")
    control.create_synapse(neuron_excite_sub, neuron_excite_main, "AMPA+NMDA")

    control.create_synapse(neuron_inhibit_main, neuron_inhibit_sub, "GABA")
    control.create_synapse(neuron_inhibit_sub, neuron_inhibit_main, "GABA")

    # --------------between excitatory and inhibitory----------------
    control.create_synapse(neuron_excite_main, neuron_inhibit_main, "AMPA+NMDA")
    control.create_synapse(neuron_inhibit_main, neuron_excite_main, "GABA")


    # ----------------output layer----------------------
    control.create_synapse(neuron_excite_main, neuron_output, "AMPA")



    # recording arrays
    voltages = []
    currents = []

    for t in range(pointCount):
        voltages_tstep = []
        currents_tstep = []
        
        if input_pattern[0,t]:
            neuron_input_0.sending_signal()
            neuron_input_0.fire_tstep.append(t)
            
        if input_pattern[1,t]:
            neuron_input_1.sending_signal()
            neuron_input_1.fire_tstep.append(t)
            
        if input_pattern[2,t]:
            neuron_input_2.sending_signal()
            neuron_input_2.fire_tstep.append(t)


        # update the synapse states then each neuron\
        num_cycle = 0
        for neuron in neurons[3:]:
            
            neuron.check_firing(t)
            neuron.update()
            
            voltages_tstep.append(neuron.Vm)
            # # only want to record the two excite main neurons
            # # they are connected to the output neuron
            if num_cycle == 4:
                currents_tstep.append(neuron.I)
            num_cycle += 1
            
        # set the synapse states back to 0
        for synapse in control.all_synapses:
            synapse.state = 0
        
        voltages.append(voltages_tstep)
        currents.append(currents_tstep)
            
    return voltages, currents, neuron_names

