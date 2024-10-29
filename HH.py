#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18

I rewrote the code from and fixed some problems and add sth new:
Python implementation of the Hodgkin-Huxley spiking neuron model
https://github.com/swharden/pyHH
@author: yaning
"""

import numpy as np


# gate for voltage gated channels 
class Gate():
    def __init__(self, alpha, beta, state):
        self.alpha = alpha
        self.beta = beta
        self.state = state
    
    # state change is the function of time 
    # basing on the alpha and beta
    def update(self, deltaTms):
        alphaState = self.alpha * (1-self.state)
        betaState = self.beta * self.state
        self.state += deltaTms * (alphaState - betaState)
    
    # def update_alpha_beta(self):

    # def setInfiniteState(self):
    #     self.state = self.alpha / (self.alpha + self.beta)


class Channel():
    def __init__(self, gMax, gP,  rE, Vm):
        self.gMax = gMax
        self.gP = gP
        self.rE = rE
        self.Vm = Vm

    def update_gP(self):
        # set the gP to 1 if it is set anotherwise
        self.gP = 1

    def current(self):
        I = self.gMax * self.gP *(self.Vm - self.rE)
        return I



class VoltageGatedChannel(Channel):
    def update_gP(self, m, n, h, deltaTms):

        # update the hyperparameters of the states: alpha and beta
        m.alpha = .1*((25-self.Vm) / (np.exp((25-self.Vm)/10)-1))
        m.beta = 4*np.exp(-self.Vm/18)
        n.alpha = .01 * ((10-self.Vm) / (np.exp((10-self.Vm)/10)-1))
        n.beta = .125*np.exp(-self.Vm/80)
        h.alpha = .07*np.exp(-self.Vm/20)
        h.beta = 1/(np.exp((30-self.Vm)/10)+1)

        # update the states
        m.update(deltaTms)
        n.update(deltaTms)
        h.update(deltaTms)

        # then implement the gP for each type of channel


class Voltage_Sodium(VoltageGatedChannel):
    def update_gP(self, m, n, h, deltaTms):
        super().update_gP(m, n, h, deltaTms)
        print(m.state)
        self.gP = np.power(m.state, 3) * h.state

class Voltage_Potassium(VoltageGatedChannel):
    def update_gP(self, m, n, h, deltaTms):
        super().update_gP(m, n, h, deltaTms)
        self.gP = np.power(n.state, 4)

class Voltage_Leak(VoltageGatedChannel):
    pass



# class LigandGatedChannel(Channel):
#     def update_gP(self, deltaTms):
#         increase_rate = -






