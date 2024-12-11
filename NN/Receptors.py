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

    def initialise(self):
        # print("this line runs")
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
        self.update_gP(0.1)
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

        # then implement the gP for each type of channel


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

    def __init__(self, gMax, gP, rE, Vm, params):
        super().__init__(gMax, gP, rE, Vm)
        self.tau_pre, self.tau_post, \
            self.tau_rec, self.tau_decay,\
             self.tau_rise, self.u_se, self.w, \
                 self.e, self.g_decay, self.g_rise, \
                     self.past_pre, self.past_post, \
                             self.learning_rate = params
    
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

    
    def update_w(self, t_step):
        self.past_post.append(t_step)
        self.w += self._w_update(self.past_pre, self.past_post, t_step)

# class only for NMDA
class NMDA(LigandGatedChannel):
    _mg = 0.01
    def update_gP(self, state, deltaTms):
        super().update_gP(state, deltaTms)
        self.gP = 1/(1+self._mg*np.exp(-0.062*self.Vm)/3.57) * self.gP


    # def update_gP(self, t_step, deltaTms):
    #     super().update_gP(t_step, deltaTms)
    #     add_mg = 1/(1+mg*np.exp(-0.062*self.Vm)/3.57) * self.gP
    #     return add_mg

class GABA(LigandGatedChannel):
    def current(self):
        return -super().current()

#---------------------------------------ligand gated channel factory----------------------------------
class LigandGatedChannelFactory:
    gP = 1
    
    gMax_AMPA = 0.0072
    # gMax_NMDA = 0.0012
    gMax_NMDA = 0.0144
    # gMax_GABA = 0.00004
    gMax_GABA = 0.04

    # those are from the paper so no inference
    rE_AMPA = 0
    rE_NMDA = 0
    rE_GABA = -70

    # set every initial value to 1
    # w_init 1 is too small for an action potential, 12 is fine
    # need to adjust those  
    w_init_AMPA = random.uniform(10.0, 14.0)
    w_init_NMDA = random.uniform(10.0, 14.0)
    w_init_GABA = random.uniform(10.0, 14.0)
    
    # should they be different for different receptors?
    e_init = 0.8
    g_decay_init = 1
    g_rise_init = 1
    tau_pre = 20
    tau_post = 10

    # no idea what values those are
    tau_rec = 1
    u_se = 1

    tau_decay_AMPA = 35
    tau_rise_AMPA = 7
    # tau_decay_NMDA = 10 works fine but the nmda current is too small
    tau_decay_NMDA = 15
    tau_rise_NMDA = 7
    tau_decay_GABA = 20 #----I made that up 
    tau_rise_GABA = 7 #----I made that up 
    
    # inference parameters
    infer_params = {
        "gMax_AMPA" : gMax_AMPA,
        "gMax_NMDA" : gMax_NMDA,
        "gMax_GABA" : gMax_GABA,

        "w_init_AMPA" : w_init_AMPA,
        "w_init_NMDA" : w_init_NMDA,
        "w_init_GABA" : w_init_GABA,
        
        "e_init" : e_init,
        "g_decay_init" : g_decay_init,
        "g_rise_init" : g_rise_init,
        "tau_pre" : tau_pre,
        "tau_post" : tau_post,

        "tau_rec" : tau_rec,
        "u_se" : u_se,

        "tau_decay_AMPA" : tau_decay_AMPA,
        "tau_rise_AMPA" : tau_rise_AMPA,
        "tau_decay_NMDA" : tau_decay_NMDA,
        "tau_rise_NMDA" : tau_rise_NMDA,
        "tau_decay_GABA" : tau_decay_GABA,
        "tau_rise_GABA" : tau_rise_GABA 
        }
    
    infer_names = ["gMax_AMPA", "gMax_NMDA", "gMax_GABA", "w_init_AMPA", "w_init_NMDA", "w_init_GABA",        
        "e_init", "g_decay_init", "g_rise_init", "tau_pre", "tau_post", "tau_rec", "u_se", "tau_decay_AMPA",
        "tau_rise_AMPA", "tau_decay_NMDA", "tau_rise_NMDA", "tau_decay_GABA", "tau_rise_GABA" ]

    learning_rate_AMPA = 0.5 #----I made that up 
    learning_rate_NMDA = 0.8 #----I made that up 
    learning_rate_GABA = 0.5 #----I made that up 

    past_pre = []
    past_post = []
    
    #tau_pre, tau_post, tau_rec, tau_decay, tau_rise, u_se,  w, e, g_decay, g_rise,
    #past_pre, past_post, learning_rate
    AMPA_params = [tau_pre, tau_post, tau_rec, tau_decay_AMPA, tau_rise_AMPA, u_se, w_init_AMPA, 
               e_init, g_decay_init, g_rise_init,
               past_pre, past_post,
               learning_rate_AMPA]

    NMDA_params = [tau_pre, tau_post, tau_rec, tau_decay_NMDA, tau_rise_NMDA, u_se, w_init_NMDA, 
               e_init, g_decay_init, g_rise_init,
               past_pre, past_post,
               learning_rate_NMDA]

    GABA_params = [tau_pre, tau_post, tau_rec, tau_decay_GABA, tau_rise_GABA, u_se, w_init_GABA, 
               e_init, g_decay_init, g_rise_init,
               past_pre, past_post,
               learning_rate_GABA]
    


    @staticmethod
    def create_AMPA(Vm=None):
        return LigandGatedChannel(LigandGatedChannelFactory.gMax_AMPA, 
                                  LigandGatedChannelFactory.gP, 
                                  LigandGatedChannelFactory.rE_AMPA, 
                                  Vm, 
                                  LigandGatedChannelFactory.AMPA_params)

    @staticmethod
    def create_NMDA(Vm=None):
        return NMDA(LigandGatedChannelFactory.gMax_NMDA, 
                                  LigandGatedChannelFactory.gP, 
                                  LigandGatedChannelFactory.rE_NMDA, 
                                  Vm, 
                                  LigandGatedChannelFactory.NMDA_params)

    @staticmethod
    def create_GABA(Vm=None):
        return GABA(LigandGatedChannelFactory.gMax_GABA, 
                                  LigandGatedChannelFactory.gP, 
                                  LigandGatedChannelFactory.rE_GABA, 
                                  Vm, 
                                  LigandGatedChannelFactory.GABA_params)








