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
    
        self._sodium_channel = Receptors.Voltage_Sodium(self.Vm)
        self._potassium_channel = Receptors.Voltage_Potassium(self.Vm)
        self._leaky_channel = Receptors.Voltage_Leak(self.Vm)
    
    # voltage and ion channel currents update has to be here because it sums from all dendrites
    def update(self):
        error_code = 0
        # Q: should i update voltage before the gP, do they make a difference?
        # update the ion channel voltage
        self._sodium_channel.Vm = self.Vm
        self._potassium_channel.Vm = self.Vm
        self._leaky_channel.Vm = self.Vm
        
        try:
            # ion channel currents
            self._sodium_channel.update_gP(self.deltaTms)
            self._potassium_channel.update_gP(self.deltaTms)
            
            # update the receptors gP in neuron
            # cannot do it in synapse because the amount of updating will depend on the connections 
            # but it shouldn't 
            Ireceptors = 0
            synapses = self.incoming_synapses
            for synapse in synapses:
                receptors = synapse.receptors
                for receptor in receptors:
                    receptor.Vm = self.Vm
                    receptor.update_gP(synapse.state, self.deltaTms)
                    Ireceptors += receptor.current()
                    
            
            
            # get the currents
            Ina = self._sodium_channel.current()
            Ik = self._potassium_channel.current()
            Ileak = self._leaky_channel.current()

            # not caring the ligand gated receptors for now

            currents = {
                "INa": Ina,
                "IK": Ik,
                "Ileak": Ileak,
                "IReceptor": Ireceptors
                # "Iampa": Iampa,
                # "Inmda": Inmda,
                # "Igaba": Igaba
            }
            # check for over or underflow
            
            for name, current in currents.items():
                if current > 1e10:
                    raise OverflowError(f"Overflowed: {name} = {current}")
        except OverflowError as m:
            # print(f"error: {m}")
            # print("this line runs")
            error_code += 1
            

        # add the ion channel currents
        self.I = Ina + Ik + Ileak + Ireceptors
        self.Vm += - self.deltaTms * self.I / self._Cm

        return error_code

    # when this neuron fires, send signal to the connected post synapses
    # this step is problematic because it only sends out signal when i am certain it fires 
    # like for now it is above -50 (the values need to substract 70)
    # and the signal is also very short lived?
    # also update the synapse weight with the pre-synapse neuron
    def check_firing(self, t):
        if self.Vm >= 20:
            self.sending_signal()
            # print(f"Neuron: {self.Name} fires")
            self.fire_tstep.append(t)
            # when post synapses fire update the weights between pre and post
            # in this function the past_post is plus one
            for synapse in self.incoming_synapses:
                for receptor in synapse.receptors:
                    receptor.update_w(t)

            # when this neuron is pre synapse then record it to past_pre
            for synapse in self.outgoing_synapses:
                for receptor in synapse.receptors:
                    receptor.past_pre.append(t)


    # change the post synapses state to active
    def sending_signal(self):
        for synapse in self.outgoing_synapses:
            synapse.state = 1


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

    


#--------------------------------Control (make connections)--------------------------------------
class Control:
    def __init__(self, deltaTms, Vm):
        self.all_synapses = []
        self.deltaTms = deltaTms
        # this voltage is the post synpase neuron voltage
        self.Vm = Vm

    # also let the neurons know their connections and keep record of all synapses so they can be updated easily
    def create_synapse(self, send_neuron, receive_neuron, type):
        
        # create receptors accordingly
        if type == "AMPA":
            # temporal solution for weight randomise
            # Receptors.LigandGatedChannelFactory.set_params()
            ampa_receptor = Receptors.LigandGatedChannelFactory.create_AMPA(self.Vm)
            synapse = Synapse(self.deltaTms, 0, send_neuron, receive_neuron, ampa_receptor)
            
        elif type == "AMPA+NMDA":
            # Receptors.LigandGatedChannelFactory.set_params()
            ampa_receptor = Receptors.LigandGatedChannelFactory.create_AMPA(self.Vm)
            nmda_receptor = Receptors.LigandGatedChannelFactory.create_NMDA(self.Vm)
            synapse = Synapse(self.deltaTms, 0, send_neuron, receive_neuron, ampa_receptor, nmda_receptor)
        
        elif type == "GABA":
            # Receptors.LigandGatedChannelFactory.set_params()
            # print(Receptors.LigandGatedChannelFactory.w_init_GABA)
            gaba_receptor = Receptors.LigandGatedChannelFactory.create_GABA(self.Vm)
            synapse = Synapse(self.deltaTms, 0, send_neuron, receive_neuron, gaba_receptor)

        send_neuron.outgoing_synapses.append(synapse)
        receive_neuron.incoming_synapses.append(synapse)

        self.all_synapses.append(synapse)

            
            