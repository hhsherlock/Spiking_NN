#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 04

@author: yaning
"""

import threshold.Receptors as Receptors
import importlib
import numpy as np
import matplotlib.pyplot as plt
import random

#--------------------------------Neuron--------------------------------------
class Neuron:
    _Cm = 1
    # choice of below two parameters see plots 
    _threshold = 12
    _duration = 61

    def __init__(self, deltaTms, I, Vm, Name):
        self.deltaTms = deltaTms
        self.I = I
        self.Vm = Vm
        self.Name = Name  
        self.fire_state = 0
        # self.fire_count = 0
        self.fire_tstep = []  

        self.incoming_synapses = []
        self.outgoing_synapses = []
        
    
        self._sodium_channel = Receptors.Voltage_Sodium(self.Vm)
        self._potassium_channel = Receptors.Voltage_Potassium(self.Vm)
        self._leaky_channel = Receptors.Voltage_Leak(self.Vm)
        
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
        # # if the neuron is not firing, update everything
        # if self.fire_state == 0:
        print("this line runs")
        # Q: should i update voltage before the gP, do they make a difference?
        # update the ion channel voltage
        self._sodium_channel.Vm = self.Vm
        self._potassium_channel.Vm = self.Vm
        self._leaky_channel.Vm = self.Vm

        # # ion channel currents
        self._sodium_channel.update_gP(self.deltaTms)
        self._potassium_channel.update_gP(self.deltaTms)
        
        # update the receptors gP in neuron
        # cannot do it in synapse because the amount of updating will depend on the connections 
        # but it shouldn't 
        # Ireceptors = 0
        # # only add receptor currents if the neuron is not firing
        # if self.Vm <= self._threshold:
        #     synapses = self.incoming_synapses
        #     for synapse in synapses:
        #         receptors = synapse.receptors
        #         for receptor in receptors:
        #             receptor.Vm = self.Vm
        #             receptor.update_gP(synapse.state, self.deltaTms)
        #             Ireceptors += receptor.current()
                

        Ina = self._sodium_channel.current()
        Ik = self._potassium_channel.current()
        Ileak = self._leaky_channel.current()

        print("Ina")
        print(np.sign(Ina))
        print("Ik")
        print(np.sign(Ik))
        print("Ileak")
        print(np.sign(Ileak))
        # print("Ireceptor")
        # print(np.sign(Ireceptors))

        # self.I = Ina + Ik + Ileak + Ireceptors
        Isum = Ina - Ik - Ileak # + Ireceptors
        # ion_I = Ina + Ik + Ileak
        self.I = Isum
        self.Vm += - self.deltaTms * self.I / self._Cm
    
        # # the fire state is 1 (fires) then only update the receptors
        # else:
        #     synapses = self.incoming_synapses
        #     for synapse in synapses:
        #         receptors = synapse.receptors
        #         for receptor in receptors:
        #             # have to update the known voltages because of this
        #             receptor.Vm = self.Vm
        #             receptor.update_gP(synapse.state, self.deltaTms)

        # # check if it should fire
        # if self.Vm >= self.fire_threshold and self.fire_count <= 61:
        #     self.fire_state = 1
        #     self.fire_count += 1
        #     self.Vm = 
        
        

        # return ion_I, Ireceptors

    # when this neuron fires, send signal to the connected post synapses
    # this step is problematic because it only sends out signal when i am certain it fires 
    # like for now it is above -50 (the values need to substract 70)
    # and the signal is also very short lived?
    # also update the synapse weight with the pre-synapse neuron
    def check_firing(self, t):
        fire = False
        if self.Vm >= self._threshold:
            self.sending_signal()
            self.fire_state = 1
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
        
            