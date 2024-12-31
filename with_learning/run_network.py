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
    
    # Neuron: deltaTms, I, Vm, fire times
    
    # with learning every group has more neurons 
    neuron_input_0 = Network.Neuron(deltaTms, 0, initial_Vm, "input_0", 0)
    neuron_input_1 = Network.Neuron(deltaTms, 0, initial_Vm, "input_1", 0)
    neuron_input_2 = Network.Neuron(deltaTms, 0, initial_Vm, "input_2", 0)

    neuron_excite_main_0 = Network.Neuron(deltaTms, 0, initial_Vm, "excite_main_0", 0)
    neuron_excite_main_1 = Network.Neuron(deltaTms, 0, initial_Vm, "excite_main_1", 0)
    neuron_excite_sub_0 = Network.Neuron(deltaTms, 0, initial_Vm, "excite_sub_0", 0)
    neuron_excite_sub_1 = Network.Neuron(deltaTms, 0, initial_Vm, "excite_sub_1", 0)

    neuron_inhibit_main_0 = Network.Neuron(deltaTms, 0, initial_Vm, "inhibit_main_0", 0)
    neuron_inhibit_main_1 = Network.Neuron(deltaTms, 0, initial_Vm, "inhibit_main_1", 0)
    neuron_inhibit_sub_0 = Network.Neuron(deltaTms, 0, initial_Vm, "inhibit_sub_0", 0)
    neuron_inhibit_sub_1 = Network.Neuron(deltaTms, 0, initial_Vm, "inhibit_sub_1", 0)

    neuron_output = Network.Neuron(deltaTms, 0, initial_Vm, "output", 0)

    neurons = [neuron_input_0, neuron_input_1, neuron_input_2, 
               neuron_excite_main_0, neuron_excite_main_1, 
               neuron_excite_sub_0, neuron_excite_sub_1, 
               neuron_inhibit_main_0, neuron_inhibit_main_1, 
               neuron_inhibit_sub_0, neuron_inhibit_sub_1,
               neuron_output]
    
    neuron_names = ["input_0", "input_1", "input_2",
                    "excite_main_0", "excite_main_1", 
                    "excite_sub_0", "excite_sub_1",
                    "inhibit_main_0", "inhibit_main_1",
                    "inhibit_sub_0", "inhibit_sub_1",
                    "output"]


    # create synapse/connection (send neuron, receive neuron)
    control = Network.Control(deltaTms, initial_Vm)
    


    #*********************full layer***************************
    # ----------------first input layer------------------------
    control.create_synapse(neuron_input_0, neuron_excite_main_0, "AMPA")
    control.create_synapse(neuron_input_1, neuron_excite_main_0, "AMPA")
    control.create_synapse(neuron_input_2, neuron_excite_main_0, "AMPA")
    
    control.create_synapse(neuron_input_0, neuron_excite_main_1, "AMPA")
    control.create_synapse(neuron_input_1, neuron_excite_main_1, "AMPA")
    control.create_synapse(neuron_input_2, neuron_excite_main_1, "AMPA")
    
    control.create_synapse(neuron_input_0, neuron_inhibit_main_0, "GABA")
    control.create_synapse(neuron_input_1, neuron_inhibit_main_0, "GABA")
    control.create_synapse(neuron_input_2, neuron_inhibit_main_0, "GABA")

    control.create_synapse(neuron_input_0, neuron_inhibit_main_1, "GABA")
    control.create_synapse(neuron_input_1, neuron_inhibit_main_1, "GABA")
    control.create_synapse(neuron_input_2, neuron_inhibit_main_1, "GABA")
    
    # ----------------self recurrent layer----------------
    control.create_synapse(neuron_excite_main_0, neuron_excite_sub_0, "AMPA+NMDA")
    control.create_synapse(neuron_excite_main_0, neuron_excite_sub_1, "AMPA+NMDA")

    control.create_synapse(neuron_excite_main_1, neuron_excite_sub_0, "AMPA+NMDA")
    control.create_synapse(neuron_excite_main_1, neuron_excite_sub_1, "AMPA+NMDA")
    
    control.create_synapse(neuron_excite_sub_0, neuron_excite_main_0, "AMPA+NMDA")
    control.create_synapse(neuron_excite_sub_0, neuron_excite_main_1, "AMPA+NMDA")

    control.create_synapse(neuron_excite_sub_1, neuron_excite_main_0, "AMPA+NMDA")
    control.create_synapse(neuron_excite_sub_1, neuron_excite_main_1, "AMPA+NMDA")
     

    control.create_synapse(neuron_inhibit_main_0, neuron_inhibit_sub_0, "GABA")
    control.create_synapse(neuron_inhibit_main_0, neuron_inhibit_sub_1, "GABA")

    control.create_synapse(neuron_inhibit_main_1, neuron_inhibit_sub_0, "GABA")
    control.create_synapse(neuron_inhibit_main_1, neuron_inhibit_sub_1, "GABA")
    
    control.create_synapse(neuron_inhibit_sub_0, neuron_inhibit_main_0, "GABA")
    control.create_synapse(neuron_inhibit_sub_0, neuron_inhibit_main_1, "GABA")

    control.create_synapse(neuron_inhibit_sub_1, neuron_inhibit_main_0, "GABA")
    control.create_synapse(neuron_inhibit_sub_1, neuron_inhibit_main_1, "GABA")
    
    # --------------between excitatory and inhibitory----------------
    control.create_synapse(neuron_excite_main_0, neuron_inhibit_main_0, "AMPA+NMDA")
    control.create_synapse(neuron_excite_main_0, neuron_inhibit_main_1, "AMPA+NMDA")

    control.create_synapse(neuron_excite_main_1, neuron_inhibit_main_0, "AMPA+NMDA")
    control.create_synapse(neuron_excite_main_1, neuron_inhibit_main_1, "AMPA+NMDA")
    
    control.create_synapse(neuron_inhibit_main_0, neuron_excite_main_0, "GABA")
    control.create_synapse(neuron_inhibit_main_0, neuron_excite_main_1, "GABA")

    control.create_synapse(neuron_inhibit_main_1, neuron_excite_main_0, "GABA")
    control.create_synapse(neuron_inhibit_main_1, neuron_excite_main_1, "GABA")

    # ----------------output layer----------------------
    control.create_synapse(neuron_excite_main_0, neuron_output, "AMPA")
    control.create_synapse(neuron_excite_main_1, neuron_output, "AMPA")


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
        # if t >= 2000 and t <= 2047:
        #     neuron_input_0.sending_signal()
            
        # if t >= 4000 and t <= 4047:
        #     neuron_input.sending_signal()
        

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
    

    

