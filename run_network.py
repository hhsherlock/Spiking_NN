#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24

@author: yaning
"""

import importlib
import numpy as np
import matplotlib.pyplot as plt

# my own class files
import NN.Receptors as Receptors
import NN.Network as Network


importlib.reload(Receptors)
importlib.reload(Network)

pointCount = 5000
deltaTms = 0.05
times = np.arange(pointCount) * deltaTms
initial_Vm = 1.3458754117369027

# everything needs to be reset including neurons 
def run(infer_params):
    Receptors.LigandGatedChannelFactory.infer_params = infer_params
    
    # Neuron: deltaTms, I, Vm
    neuron_input = Network.Neuron(deltaTms, 0, initial_Vm, "input", 0)

    neuron_excite_main = Network.Neuron(deltaTms, 0, initial_Vm, "excite_main", 0)
    neuron_excite_sub = Network.Neuron(deltaTms, 0, initial_Vm, "excite_sub", 0)

    neuron_inhibit_main = Network.Neuron(deltaTms, 0, initial_Vm, "inhibit_main", 0)
    neuron_inhibit_sub = Network.Neuron(deltaTms, 0, initial_Vm, "inhibit_sub", 0)

    neuron_output = Network.Neuron(deltaTms, 0, initial_Vm, "output", 0)

    neurons = [neuron_input, neuron_excite_main, neuron_excite_sub, 
            neuron_inhibit_main, neuron_inhibit_sub, neuron_output]
    
    neuron_names = ["input", "excite_main", "excite_sub", "inhibit_main", "inhibit_sub", "output"]





    # create synapse/connection (send neuron, receive neuron)
    control = Network.Control(deltaTms, initial_Vm)
    



    #*********************full layer***************************
    # ----------------first input layer------------------------
    control.create_synapse(neuron_input, neuron_excite_main, "AMPA")
    control.create_synapse(neuron_input, neuron_inhibit_main, "GABA")

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
    neuron_voltages = []
    neuron_firing = []
    neuron_error = []


    # run
    for t in range(pointCount):
        # record every time step is a list
        voltage_Tstep = []
        fire_Tstep = []
        error_code_Tstep = []

        # simulate input neuron firing
        # this step changes states of the receptors
        if t >= 2000 and t <= 2500:
            neuron_input.sending_signal()
        

        # update the synapse states then update each neuron
        for neuron in neurons:
            neuron.check_firing()
            error_code = neuron.update()

            # record 
            voltage_Tstep.append(neuron.Vm)
            fire_Tstep.append(neuron.fire_times)
            error_code_Tstep.append(error_code)
        
            
        # set the synapse states back to 0
        for synapse in control.all_synapses:
            synapse.state = 0
            

        neuron_voltages.append(voltage_Tstep)
        neuron_firing.append(fire_Tstep)
        neuron_error.append(error_code_Tstep)
    return neuron_voltages, neuron_firing, neuron_error, neuron_names
    

    

