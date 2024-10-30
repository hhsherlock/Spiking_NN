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



class LigandGatedChannel(Channel):

    def __init__(self, gMax, gP, rE, Vm, params):
        super().__init__(gMax, gP, rE, Vm)
        self.tau_pre, self.tau_rec, self.tau_decay,\
             self.tau_rise, self.u_se, self.w, \
                 self.e, self.g_decay, self.g_rise, \
                     self.past_pre, self.past_post, \
                         self.tsp_pre, self.tsp_post, \
                             self.learning_rate = params
    
    def _runge_kutta(self, f, y0, h, *arg):
        k1 = f(y0, *arg)
        k2 = f(y0 + h*k1/2, *arg)
        k3 = f(y0 + h*k2/2, *arg)
        k4 = f(y0 + h*k3, *arg)

        e_next = y0 + 1/6*(k1 + 2*k2 + 2*k3 + k4)

        return e_next    
    
    # integrate function for weight
    def _integrate(self, past, current):
        integrate_result = 0
        for i in past:
            integrate_result += np.exp(-(current-i)/self.tau_pre)
        return integrate_result
    
    # w(m,n) synapse weight
    def _w_update(self, past_pre, past_post, t_step):
        return self.learning_rate*self._integrate(
            past_pre, t_step)*self._integrate(past_post, t_step)
    
    # e(m,n) synaptic efficacy
    def _e_update(self, e, etsp):
        return (1-e)/self.tau_rec - self.u_se*etsp

    # G decay and rise
    def _g_decay_update(self, g_decay, w, e):
        return -g_decay/self.tau_decay + w*e

    def _g_rise_update(self, g_rise, w, e):
        return -g_rise/self.tau_rise + w*e

    def update_gP(self, t_step, deltaTms):
            # updating e value
        if t_step not in self.tsp_pre:
            self.e = self._runge_kutta(self._e_update, self.e, deltaTms, 0)
        else:
            self.e = self._runge_kutta(self._e_update, self.e, deltaTms, self.e)
            self.past_pre.append(t_step)

        # updating w value
        if t_step in self.tsp_post:
            self.past_post.append(t_step)
            self.w += self._w_update(self.past_pre, self.past_post, t_step)

        # updating g_decay based on e and w
        if t_step not in self.tsp_pre:
            self.g_decay = self._runge_kutta(self._g_decay_update, self.g_decay, deltaTms, 0,0)
            self.g_rise = self._runge_kutta(self._g_rise_update, self.g_rise, deltaTms, 0, 0)
        else:
            self.g_decay = self._runge_kutta(self._g_decay_update, self.g_decay, deltaTms, self.w, self.e)
            self.g_rise = self._runge_kutta(self._g_rise_update, self.g_rise, deltaTms, self.w, self.e)

        self.gP = self.g_rise - self.g_decay









