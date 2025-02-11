#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 04

@author: yaning
"""

import with_learning.learning_NN.Receptors as Receptors
import importlib
import numpy as np
import matplotlib.pyplot as plt
import random

#--------------------------------Neuron--------------------------------------
class Neuron:
    _Cm = 1

    def __init__(self, deltaTms, I, Vm, Name):
        self.deltaTms = deltaTms
        self.I = I
        self.Vm = Vm
        self.Name = Name      
        self.fire_tstep = []  

        self.incoming_synapses = []
        self.outgoing_synapses = []
        
        # # for threshold we don't need ion channels
        # self._sodium_channel = Receptors.Voltage_Sodium(self.Vm)
        # self._potassium_channel = Receptors.Voltage_Potassium(self.Vm)
        # self._leaky_channel = Receptors.Voltage_Leak(self.Vm)
        
    @staticmethod
    def count_all_continuous_sequences(arr):
        if not arr:
            return [0]

        # Sort the array to ensure it's in order
        arr = sorted(arr)

        new_arr = []
        new_arr.append(arr[0])
        # Iterate through the array
        for i in range(1, len(arr)):
            if arr[i] != arr[i - 1] + 1:
                # A break in continuity means a new sequence
                new_arr.append(arr[i])

        return new_arr
    
    # voltage and ion channel currents update has to be here because it sums from all dendrites
    def update(self):

        # Q: should i update voltage before the gP, do they make a difference?
        # update the ion channel voltage
        # self._sodium_channel.Vm = self.Vm
        # self._potassium_channel.Vm = self.Vm
        # self._leaky_channel.Vm = self.Vm

        Ireceptors = 0
        synapses = self.incoming_synapses
        for synapse in synapses:
            receptors = synapse.receptors
            for receptor in receptors:
                receptor.Vm = self.Vm
                receptor.update_gP(synapse.state, self.deltaTms)
                Ireceptors += receptor.current()
                    
            

        # add the ion channel currents, what about the ion channel currents
        self.I = Ireceptors
        self.Vm += - self.deltaTms * self.I / self._Cm


    # when this neuron fires, send signal to the connected post synapses
    # this step is problematic because it only sends out signal when i am certain it fires 
    # like for now it is above -50 (the values need to substract 70)
    # and the signal is also very short lived?
    # also update the synapse weight with the pre-synapse neuron
    def check_firing(self, t):
        fire = False
        if self.Vm >= 20:
            self.sending_signal()
            # print(f"Neuron: {self.Name} fires")
            self.fire_tstep.append(t)
            fire = True
        return fire

    def update_weights(self, t):
        for synapse in self.incoming_synapses:
            neuron_past_pre = self.count_all_continuous_sequences(synapse.send_neuron.fire_tstep)
            neuron_past_post = self.count_all_continuous_sequences(self.fire_tstep)
            for receptor in synapse.receptors:
                receptor.update_w(t, neuron_past_pre, neuron_past_post)
                    



    # change the post synapses state to active
    def sending_signal(self):
        for synapse in self.outgoing_synapses:
            random_value = random.uniform(0, 1)
            if random_value >= 0.4:
                synapse.state = 1
    
    def erase(self, initial_Vm):
        self.I = 0
        self.Vm = initial_Vm   
        self.fire_tstep = []  
    
        self._sodium_channel = Receptors.Voltage_Sodium(initial_Vm)
        self._potassium_channel = Receptors.Voltage_Potassium(initial_Vm)
        self._leaky_channel = Receptors.Voltage_Leak(initial_Vm)


#--------------------------------Synapse--------------------------------------
# ligand gated receptors belong to the synapse
class Synapse:
    def __init__(self, deltaTms, state, send_neuron, receive_neuron, *args):
        # either it is being simulated 1 or not 0
        # need the state to update gP differently
        self.deltaTms = deltaTms
        self.state = state

        # do we need send neurons?
        self.send_neuron = send_neuron
        self.receive_neuron = receive_neuron

        # the type of connection
        self.receptors = []
        if args:
            self.receptors.extend(args)
          


# #--------------------------------Control (make connections)--------------------------------------
# class Control:
#     def __init__(self, deltaTms, Vm):
#         self.all_synapses = []
#         self.deltaTms = deltaTms
#         # this voltage is the post synpase neuron voltage
#         self.Vm = Vm

#     # also let the neurons know their connections and keep record of all synapses so they can be updated easily
#     def create_synapse(self, send_neuron, receive_neuron, type):
        
#         # create receptors accordingly
#         if type == "AMPA":
#             # temporal solution for weight randomise
#             # Receptors.LigandGatedChannelFactory.set_params()
#             ampa_receptor = Receptors.AMPA(0.00072, 1, -70, self.Vm, 0.9, 1, 1, 1, 12, 10, 20, 10, 35, 7, 0.7, "AMPA")
#             synapse = Synapse(self.deltaTms, 0, send_neuron, receive_neuron, ampa_receptor)
            
#         elif type == "AMPA+NMDA":
#             # Receptors.LigandGatedChannelFactory.set_params()
#             ampa_receptor = Receptors.AMPA(0.00072, 1, -70, self.Vm, 0.9, 1, 1, 1, 12, 10, 20, 10, 35, 7, 0.7, "AMPA")
#             nmda_receptor = Receptors.NMDA(0.0012, 1, -70, self.Vm, 0.9, 1, 1, 1, 12, 10, 20, 10, 15, 7, 0.7, "NMDA")
#             synapse = Synapse(self.deltaTms, 0, send_neuron, receive_neuron, ampa_receptor, nmda_receptor)
        
#         elif type == "GABA":
#             # Receptors.LigandGatedChannelFactory.set_params()
#             # print(Receptors.LigandGatedChannelFactory.w_init_GABA)
#             gaba_receptor = Receptors.GABA(0.0012, 1, -140, self.Vm, 0.9, 1, 1, 1, 12, 10, 20, 10, 20, 7, 0.7, "GABA")
#             synapse = Synapse(self.deltaTms, 0, send_neuron, receive_neuron, gaba_receptor)

#         send_neuron.outgoing_synapses.append(synapse)
#         receive_neuron.incoming_synapses.append(synapse)

#         self.all_synapses.append(synapse)

#     def update_synapse_initial_values(self, infer_params):
#         for synapse in self.all_synapses:
#             for receptor in synapse.receptors:
#                 receptor.e = infer_params["e_init"]
#                 receptor.u_se = infer_params["u_se"]
#                 receptor.g_decay = infer_params["g_decay_init"]
#                 receptor.g_rise = infer_params["g_rise_init"]
#                 receptor.w = infer_params["w_init"]
#                 receptor.tau_rec = infer_params["tau_rec"]
#                 receptor.tau_pre = infer_params["tau_pre"]
#                 receptor.tau_post = infer_params["tau_post"]

#                 if receptor.label == "GABA":
#                     receptor.gMax = infer_params["gMax_GABA"]
#                     receptor.tau_decay = infer_params["tau_decay_GABA"]
#                     receptor.tau_rise = infer_params["tau_rise_GABA"]
                
#                 elif receptor.label == "NMDA":
#                     receptor.tau_decay = infer_params["tau_decay_NMDA"]
#                     receptor.tau_rise = infer_params["tau_rise_NMDA"]
                
#                 elif receptor.label == "AMPA":
#                     receptor.tau_decay = infer_params["tau_decay_AMPA"]
#                     receptor.tau_rise = infer_params["tau_rise_AMPA"]

        
            