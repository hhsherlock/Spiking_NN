#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 04

@author: yaning
"""

import weight_normalise.Receptors as Receptors
import importlib
import numpy as np
import matplotlib.pyplot as plt
import random

#--------------------------------Neuron--------------------------------------
class Neuron:
    _Cm = 1
    _threshold = -50 + 70
    _current_threshold = -200
    _w_increase = 10
    _w_decrease = 10

    def __init__(self, deltaTms, I, Vm, Name):
        self.deltaTms = deltaTms
        self.I = I
        self.Vm = Vm
        self.Name = Name  

        self.num = 0

        self.fire_tstep = []  

        self.incoming_synapses = []
        self.outgoing_synapses = []
        
    
        self._sodium_channel = Receptors.Voltage_Sodium(self.Vm)
        self._potassium_channel = Receptors.Voltage_Potassium(self.Vm)
        self._leaky_channel = Receptors.Voltage_Leak(self.Vm)
        
    
 
    def update(self):

        # update ion channels
        self._sodium_channel.Vm = self.Vm
        self._potassium_channel.Vm = self.Vm
        self._leaky_channel.Vm = self.Vm

        self._sodium_channel.update_gP(self.deltaTms)
        self._potassium_channel.update_gP(self.deltaTms)
        
        # update receptors
        Ireceptors = 0
        synapses = self.incoming_synapses
        for synapse in synapses:
            receptors = synapse.receptors
            for receptor in receptors:
                receptor.Vm = self.Vm
                receptor.update_gP(synapse.state, self.deltaTms)
                Ireceptors += receptor.current()
        
        # receptor current should always be negative
        if Ireceptors > 0:
            Ireceptors = 0 
        
        # if the receptor current is too big 
        # decrease the ampa and nmda weight and increase gaba weight 
        if Ireceptors <= self._current_threshold:
            self.num += 1
            for synapse in self.incoming_synapses:
                for receptor in synapse.receptors:
                    if receptor.label == "GABA":
                        # or should i change it proportional
                        receptor.w = receptor.w*1.3
                    else:
                        receptor.w = receptor.w*0.7


        Ina = self._sodium_channel.current()
        Ik = self._potassium_channel.current()
        Ileak = self._leaky_channel.current()

        

        self.I = - Ina - Ik - Ileak - Ireceptors
        self.Vm += self.deltaTms * self.I / self._Cm
          

        return Ireceptors, Ina

    # when this neuron fires, send signal to the connected post synapses
    # this step is problematic because it only sends out signal when i am certain it fires 
    # like for now it is above -50 (the values need to substract 70)
    # and the signal is also very short lived?
    # also update the synapse weight with the pre-synapse neuron
    def check_firing(self, t):
        fire = False
        if self.Vm >= self._threshold:
            self.sending_signal()
            # print(f"Neuron: {self.Name} fires")
            self.fire_tstep.append(t)
            fire = True
        return fire

    # two ways to update weight/learn
    def update_weights_hebbian(self, t):
        for synapse in self.incoming_synapses:
            #Q: should it be the whole process or a time frame
            neuron_past_pre = count_all_continuous_sequences(synapse.send_neuron.fire_tstep)
            neuron_past_post = count_all_continuous_sequences(self.fire_tstep)
            for receptor in synapse.receptors:
                receptor.update_w_hebbian(t, neuron_past_pre, neuron_past_post)

    def update_weights_twenty(self, t):
        #[t-400:t+401]

        for synapse in self.incoming_synapses:
            pre_neuron = count_all_continuous_sequences(synapse.send_neuron.fire_tstep[t-400:])
            if pre_neuron:
                for receptor in synapse.receptors:
                    receptor.update_w_twenty(t, pre_neuron, True)
        
        for synapse in self.outgoing_synapses:
            post_neuron = count_all_continuous_sequences(synapse.receive_neuron.fire_tstep[t-400:])
            if post_neuron:
                for receptor in synapse.receptors:
                    receptor.update_w_twenty(t, post_neuron, False)

            






    # change the post synapses state to active
    def sending_signal(self):
        for synapse in self.outgoing_synapses:
            random_value = random.uniform(0, 1)
            if random_value >= 0.3:
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