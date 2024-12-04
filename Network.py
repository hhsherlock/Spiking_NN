#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 04

@author: yaning
"""

import Receptors as Receptors
import importlib
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------Neuron--------------------------------------
class Neuron:
    _Cm = 1

    def __init__(self, deltaTms, I, Vm):
        self.deltaTms = deltaTms
        self.I = I
        self.Vm = Vm
        self.incoming_synapses = []
        self.outgoing_synapses = []
    
        self._sodium_channel = Receptors.Voltage_Sodium(self.Vm)
        self._potassium_channel = Receptors.Voltage_Potassium(self.Vm)
        self._leaky_channel = Receptors.Voltage_Leak(self.Vm)
    
    # voltage and ion channel currents update has to be here because it sums from all dendrites
    def update(self):
        # ion channel currents
        self._sodium_channel.update_gP(self.deltaTms)
        self._potassium_channel.update_gP(self.deltaTms)
        

        try:
        # get the currents
            Ina = self._sodium_channel.current()
            Ik = self._potassium_channel.current()
            Ileak = self._leaky_channel.current()

            # not caring the ligand gated receptors for now

            currents = {
                "INa": Ina,
                "IK": Ik,
                "Ileak": Ileak,
                # "Iampa": Iampa,
                # "Inmda": Inmda,
                # "Igaba": Igaba
            }
            # check for over or underflow
            
            for name, current in currents.items():
                if current > 1e10:
                    raise OverflowError(f"Overflowed: {name} = {current}")
        except OverflowError as m:
            print(f"error: {m}")
            

        # add the ion channel currents
        self.I = Ina + Ik + Ileak + self.I
        self.Vm += - self.deltaTms * self.I / self._Cm

        # update the ion channel voltage
        self._sodium_channel.Vm = self.Vm
        self._potassium_channel.Vm = self.Vm
        self._leaky_channel.Vm = self.Vm
        # print(self.Vm)
    
    # when this neuron fires, send signal to the connected post synapses
    def sending_signal(self):
        for synapse in self.outgoing_synapses:
            synapse.state = 1

    # when the pre synapses fire, it activates this function
    def inject_current(self, sum_currents):
        self.I = sum_currents

        
    # def add_incoming_synapses(self, ):
    #     self.incoming_synapses.append()

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
    
    # update the gP
    def update(self):
        sum_currents = 0
        Vm = self.receive_neuron.Vm
        for receptor in self.receptors:
            receptor.update_gP(self.state, self.deltaTms)
            sum_currents += receptor.current()

            receptor.Vm = Vm

            # print(sum_currents)
        
        self.receive_neuron.inject_current(sum_currents)

#--------------------------------Control (make connections)--------------------------------------
class Control:
    def __init__(self, deltaTms, Vm):
        self.all_synapses = []
        self.deltaTms = deltaTms
        # this voltage is the post synpase neuron voltage
        self.Vm = Vm

    # also let the neurons know their connections and keep record of all synapses so they can be updated easily
    def create_synapse(self, send_neuron, receive_neuron, type):
        
        if type == "AMPA":
            # temporal solution for weight randomise
            Receptors.LigandGatedChannelFactory.set_params()
            ampa_receptor = Receptors.LigandGatedChannelFactory.create_AMPA(self.Vm)
            synapse = Synapse(self.deltaTms, 0, send_neuron, receive_neuron, ampa_receptor)
            
        elif type == "AMPA+NMDA":
            Receptors.LigandGatedChannelFactory.set_params()
            ampa_receptor = Receptors.LigandGatedChannelFactory.create_AMPA(self.Vm)
            nmda_receptor = Receptors.LigandGatedChannelFactory.create_NMDA(self.Vm)
            synapse = Synapse(self.deltaTms, 0, send_neuron, receive_neuron, ampa_receptor, nmda_receptor)
        
        elif type == "GABA":
            Receptors.LigandGatedChannelFactory.set_params()
            gaba_receptor = Receptors.LigandGatedChannelFactory.create_GABA(self.Vm)
            synapse = Synapse(self.deltaTms, 0, send_neuron, receive_neuron, gaba_receptor)

        send_neuron.outgoing_synapses.append(synapse)
        receive_neuron.incoming_synapses.append(synapse)

        self.all_synapses.append(synapse)
            
            