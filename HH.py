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
class Gate:
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
    gMax_Na = 120
    rE_Na = 115
    gP = 1

    def __init__(self, Vm):
        super().__init__(Voltage_Sodium.gMax_Na, Voltage_Sodium.gP, Voltage_Sodium.rE_Na, Vm)

    def update_gP(self, m, n, h, deltaTms):
        super().update_gP(m, n, h, deltaTms)
        # print(m.state)
        self.gP = np.power(m.state, 3) * h.state

class Voltage_Potassium(VoltageGatedChannel):
    gMax_K = 26
    rE_K = -12
    gP = 1

    def __init__(self, Vm):
        super().__init__(Voltage_Potassium.gMax_K, Voltage_Potassium.gP, Voltage_Potassium.rE_K, Vm)
    
    def update_gP(self, m, n, h, deltaTms):
        super().update_gP(m, n, h, deltaTms)
        self.gP = np.power(n.state, 4)

class Voltage_Leak(VoltageGatedChannel):
    gMax_leaky = 0.3
    rE_leaky = 10.6
    gP = 1

    def __init__(self, Vm):
        super().__init__(Voltage_Leak.gMax_leaky, Voltage_Leak.gP, Voltage_Leak.rE_leaky, Vm)

    

#---------------------------------------ligand gated channel-------------------------------------

class LigandGatedChannel(Channel):

    def __init__(self, gMax, gP, rE, Vm, params, record_lists):
        super().__init__(gMax, gP, rE, Vm)
        self.tau_pre, self.tau_post, \
            self.tau_rec, self.tau_decay,\
             self.tau_rise, self.u_se, self.w, \
                 self.e, self.g_decay, self.g_rise, \
                     self.past_pre, self.past_post, \
                             self.learning_rate = params
        self.past_pre, self.past_post = record_lists
    
    #------------tool methods--------------------
    @staticmethod
    def _runge_kutta(f, y0, h, *arg):
        k1 = f(y0, *arg)
        k2 = f(y0 + h*k1/2, *arg)
        k3 = f(y0 + h*k2/2, *arg)
        k4 = f(y0 + h*k3, *arg)

        e_next = y0 + 1/6*(k1 + 2*k2 + 2*k3 + k4)

        return e_next    
    
    # integrate function for weight
    def _integrate(self, past, current, tau_p):
        integrate_result = 0
        for i in past:
            integrate_result += np.exp(-(current-i)/tau_p)
        return integrate_result
    
    # ----------w(m,n) synapse weight----------------
    def _w_update(self, past_pre, past_post, t_step):
        return self.learning_rate*self._integrate(
            past_pre, t_step, self.tau_pre)*self._integrate(past_post, t_step, self.tau_post)
    
    #---------e(m,n) synaptic efficacy---------------
    def _e_update(self, e, etsp):
        return (1-e)/self.tau_rec - self.u_se*etsp

    #------------G decay and rise--------------------
    def _g_decay_update(self, g_decay, w, e):
        return -g_decay/self.tau_decay + w*e

    def _g_rise_update(self, g_rise, w, e):
        return -g_rise/self.tau_rise + w*e

    #------------overwrite the update gP------------
    def update_gP(self, t_step, deltaTms, tsp_pre, tsp_post):
        # updating e value
        if t_step not in tsp_pre:
            self.e = self._runge_kutta(self._e_update, self.e, deltaTms, 0)
        else:
            self.e = self._runge_kutta(self._e_update, self.e, deltaTms, self.e)
            self.past_pre.append(t_step)

        # updating w value
        if t_step in tsp_post:
            self.past_post.append(t_step)
            self.w += self._w_update(self.past_pre, self.past_post, t_step)

        # updating g_decay based on e and w
        if t_step not in tsp_pre:
            self.g_decay = self._runge_kutta(self._g_decay_update, self.g_decay, deltaTms, 0,0)
            self.g_rise = self._runge_kutta(self._g_rise_update, self.g_rise, deltaTms, 0, 0)
        else:
            self.g_decay = self._runge_kutta(self._g_decay_update, self.g_decay, deltaTms, self.w, self.e)
            self.g_rise = self._runge_kutta(self._g_rise_update, self.g_rise, deltaTms, self.w, self.e)

        self.gP = self.g_rise - self.g_decay

class LigandGatedChannelFactory:
    gP = 1
    
    gMax_AMPA = 0.00072
    gMax_NMDA = 0.0012
    gMax_GABA = 0.00004
    rE_AMPA = 0
    rE_NMDA = 0
    rE_GABA = -70

    # set every initial value to 1
    w_init = 1
    e_init = 1
    g_decay_init = 1
    g_rise_init = 1
    tau_pre = 20
    tau_post = 10

    # no idea what values those are
    tau_rec = 1
    u_se = 1

    tau_decay_AMPA = 1.5
    tau_rise_AMPA = 0.09
    tau_decay_NMDA = 40
    tau_rise_NMDA = 3
    tau_decay_GABA = 5 #----I made that up 
    tau_rise_GABA = 5 #----I made that up 

    learning_rate_AMPA = 0.5 #----I made that up 
    learning_rate_NMDA = 0.8 #----I made that up 
    learning_rate_GABA = 0.5 #----I made that up 

    past_pre = []
    past_post = []
    
    #tau_pre, tau_post, tau_rec, tau_decay, tau_rise, u_se,  w, e, g_decay, g_rise,
    #past_pre, past_post, learning_rate
    AMPA_params = [tau_pre, tau_post, tau_rec, tau_decay_AMPA, tau_rise_AMPA, u_se, w_init, 
               e_init, g_decay_init, g_rise_init,
               past_pre, past_post,
               learning_rate_AMPA]

    NMDA_params = [tau_pre, tau_post, tau_rec, tau_decay_NMDA, tau_rise_NMDA, u_se, w_init, 
               e_init, g_decay_init, g_rise_init,
               past_pre, past_post,
               learning_rate_NMDA]

    GABA_params = [tau_pre, tau_post, tau_rec, tau_decay_GABA, tau_rise_GABA, u_se, w_init, 
               e_init, g_decay_init, g_rise_init,
               past_pre, past_post,
               learning_rate_GABA]
    
    @staticmethod
    def create_AMPA(Vm=None, record_lists = []):
        return LigandGatedChannel(LigandGatedChannelFactory.gMax_AMPA, 
                                  LigandGatedChannelFactory.gP, 
                                  LigandGatedChannelFactory.rE_AMPA, 
                                  Vm, 
                                  LigandGatedChannelFactory.AMPA_params,
                                  record_lists)

    @staticmethod
    def create_NMDA(Vm=None, record_lists = []):
        return LigandGatedChannel(LigandGatedChannelFactory.gMax_NMDA, 
                                  LigandGatedChannelFactory.gP, 
                                  LigandGatedChannelFactory.rE_NMDA, 
                                  Vm, 
                                  LigandGatedChannelFactory.NMDA_params,
                                  record_lists)

    @staticmethod
    def create_GABA(Vm=None, record_lists = []):
        return LigandGatedChannel(LigandGatedChannelFactory.gMax_GABA, 
                                  LigandGatedChannelFactory.gP, 
                                  LigandGatedChannelFactory.rE_GABA, 
                                  Vm, 
                                  LigandGatedChannelFactory.NMDA_params,
                                  record_lists)








