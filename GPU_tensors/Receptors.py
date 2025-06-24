#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27

I rewrote the code from and fixed some problems and add sth new:
Python implementation of the Hodgkin-Huxley spiking neuron model
https://github.com/swharden/pyHH

changed stuff to run in a neural network
@author: yaning
"""

import numpy as np
import random 
import warnings


# gate for voltage gated channels 
class Gate:
    def __init__(self, alpha, beta, state):
        self.alpha = alpha
        self.beta = beta
        self.state = state
    

    def update(self, deltaTms):
        alphaState = self.alpha * (1-self.state)
        betaState = self.beta * self.state
        self.state += deltaTms * (alphaState - betaState)


    def initialise(self):
        self.state = self.alpha / (self.alpha + self.beta)


class Channel:
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

#---------------------------------------voltage gated channel-------------------------------------

class VoltageGatedChannel(Channel):
    def __init__(self, gMax, gP, rE, Vm):
        super().__init__(gMax, gP, rE, Vm)
        self.m = Gate(0,0,0)
        self.n = Gate(0,0,0)
        self.h = Gate(0,0,0)
        # this deltaTms should it be the same as the experiment?
        # self.update_alpha_beta()
        self.update_gP(0.05)
        self.m.initialise()
        self.n.initialise()
        self.h.initialise()
        # print(self.m.alpha)


    def update_gP(self, deltaTms):

        # update the hyperparameters of the states: alpha and beta
        self.m.alpha = .1*((25-self.Vm) / (np.exp((25-self.Vm)/10)-1))
        self.m.beta = 4*np.exp(-self.Vm/18)
        self.n.alpha = .01 * ((10-self.Vm) / (np.exp((10-self.Vm)/10)-1))
        self.n.beta = .125*np.exp(-self.Vm/80)
        self.h.alpha = .07*np.exp(-self.Vm/20)
        self.h.beta = 1/(np.exp((30-self.Vm)/10)+1)

        # update the states 
        self.m.update(deltaTms)
        self.n.update(deltaTms)
        self.h.update(deltaTms)



class Voltage_Sodium(VoltageGatedChannel):
    gMax_Na = 120
    rE_Na = 115
    gP = 1

    def __init__(self, Vm):
        super().__init__(Voltage_Sodium.gMax_Na, Voltage_Sodium.gP, Voltage_Sodium.rE_Na, Vm)

    def update_gP(self, deltaTms):
        super().update_gP(deltaTms)
        self.gP = np.power(self.m.state, 3) * self.h.state
    

class Voltage_Potassium(VoltageGatedChannel):
    gMax_K = 36
    rE_K = -12
    gP = 1

    def __init__(self, Vm):
        super().__init__(Voltage_Potassium.gMax_K, Voltage_Potassium.gP, Voltage_Potassium.rE_K, Vm)
    
    def update_gP(self, deltaTms):
        super().update_gP(deltaTms)
        self.gP = np.power(self.n.state, 4)
    

class Voltage_Leak(VoltageGatedChannel):
    gMax_leaky = 0.3
    rE_leaky = 10.6
    gP = 1

    def __init__(self, Vm):
        super().__init__(Voltage_Leak.gMax_leaky, Voltage_Leak.gP, Voltage_Leak.rE_leaky, Vm)

    

#---------------------------------------ligand gated channel-------------------------------------

class LigandGatedChannel(Channel):

    def __init__(self, gMax, gP, rE, Vm, e, u_se, g_decay, g_rise,
                 w, tau_rec, tau_pre, tau_post, tau_decay, tau_rise, 
                 learning_rate, label):
        super().__init__(gMax, gP, rE, Vm)
        self.e = e
        self.u_se = u_se
        self.g_decay = g_decay
        self.g_rise = g_rise
        self.w = w
        self.tau_rec = tau_rec
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.tau_decay = tau_decay
        self.tau_rise = tau_rise
        
        self.learning_rate = learning_rate
        self.label = label
    
    #------------tool methods--------------------
    @staticmethod
    def _runge_kutta(f, y0, h, *arg):
        k1 = f(y0, *arg)
        k2 = f(y0 + h*k1/2, *arg)
        k3 = f(y0 + h*k2/2, *arg)
        k4 = f(y0 + h*k3, *arg)

        next = y0 + 1/6*(k1 + 2*k2 + 2*k3 + k4)

        return next    
    
    # integrate function for weight
    def _integrate(self, past, current, tau_p):
        integrate_result = 0
        # print(past)
        for i in past:
            integrate_result += np.exp(-(current-i)*0.0005/tau_p)
        return integrate_result
    
    # ----------w(m,n) synapse weight----------------
    def _w_update_hebbian(self, past_pre, past_post, t_step):
        return self.learning_rate*self._integrate(
            past_pre, t_step, self.tau_pre)*self._integrate(past_post, t_step, self.tau_post)
    
    # i do not know what time constant this should be
    def _w_update_twenty(self, the_other_neuron, t_step):
        temp = self.learning_rate*self._integrate(
            the_other_neuron, t_step, self.tau_pre
        )
        return temp
    
    #---------e(m,n) synaptic efficacy---------------
    def _e_update(self, e, etsp):
        return (1-e)/self.tau_rec - self.u_se*etsp

    #------------G decay and rise--------------------
    def _g_decay_update(self, g_decay, w, e):
        return -g_decay/self.tau_decay + w*e

    def _g_rise_update(self, g_rise, w, e):

        return -g_rise/self.tau_rise + w*e

    #------------overwrite the update gP------------
    def update_gP(self, state, deltaTms):
        # updating e value
        if state:
            self.e = self._runge_kutta(self._e_update, self.e, deltaTms, self.e)
            # i also need to record the time steps
            
            # self.past_pre.append(t_step)

        else:
            self.e = self._runge_kutta(self._e_update, self.e, deltaTms, 0)
            # print("this line runs")
            

        # updating g_decay based on e and w
        if state:
            self.g_decay = self._runge_kutta(self._g_decay_update, self.g_decay, deltaTms*10, self.w, self.e)
            self.g_rise = self._runge_kutta(self._g_rise_update, self.g_rise, deltaTms*10, self.w, self.e)
        else:
            self.g_decay = self._runge_kutta(self._g_decay_update, self.g_decay, deltaTms*10, 0,0)
            self.g_rise = self._runge_kutta(self._g_rise_update, self.g_rise, deltaTms*10, 0, 0)
            

        self.gP = self.g_rise - self.g_decay

    
    def update_w_hebbian(self, t_step, neuron_past_pre, neuron_past_post):
        self.w += self._w_update_hebbian(neuron_past_pre, neuron_past_post, t_step)

    def update_w_twenty(self, t_step, the_other_neuron, pre_or_post):
        if pre_or_post:
            self.w += self._w_update_twenty(the_other_neuron, t_step)
        else:
            self.w -= self._w_update_twenty(the_other_neuron, t_step)
            if self.w < 0:
                self.w = 0
            
class AMPA(LigandGatedChannel):
    pass
        

# class only for NMDA
class NMDA(LigandGatedChannel):
    _mg = 0.01
    def update_gP(self, state, deltaTms):
        super().update_gP(state, deltaTms)
        self.gP = 1/(1+self._mg*np.exp(-0.062*self.Vm)/3.57) * self.gP


class GABA(LigandGatedChannel):
    def current(self):
        return -super().current()
