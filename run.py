#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24

@author: yaning
"""

import HH as HH
import importlib
import numpy as np
import matplotlib.pyplot as plt

def run(params):
    # simulation environment
    deltaTms = 0.05
    Cm = 1
    # other Vm initialisation can lead to random firing
    Vm = 1.3458754117369027
    # 5000 cycles and every cycle is 0.01ms
    # in total 50ms
    pointCount = 5000
    voltages = np.empty(pointCount)
    times = np.arange(pointCount) * deltaTms #record the actual time
    stim = np.zeros(pointCount)
    stim[1500:4000] = 20  # create a square pulse

    # presynapse firing, for now has no meaning
    tsp_pre = [500,2000,3500]


    sodium_channel = HH.Voltage_Sodium(Vm)
    potassium_channel = HH.Voltage_Potassium(Vm)
    leaky_channel = HH.Voltage_Leak(Vm)

    # because I am using factory, so the initialised values stay the same
    # even the below line runs again
    HH.LigandGatedChannelFactory.set_params(params)
    ampa_receptor = HH.LigandGatedChannelFactory.create_AMPA(Vm)

    na_currents = []
    k_currents = []
    leak_currents = []
    ampa_currents = []

    include_ampa = 1

    for i in range(len(times)):

        sodium_channel.update_gP(deltaTms)
        potassium_channel.update_gP(deltaTms)

        if include_ampa:
            ampa_receptor.update_gP(i, deltaTms, tsp_pre)


        # get the currents
        INa = sodium_channel.current()
        IK = potassium_channel.current()
        Ileak = leaky_channel.current()

        if include_ampa:
            Iampa = ampa_receptor.current()
            ampa_currents.append(Iampa)

        try:
            # get the currents
            INa = sodium_channel.current()
            IK = potassium_channel.current()
            Ileak = leaky_channel.current()

            if include_ampa:
                Iampa = ampa_receptor.current()
                ampa_currents.append(Iampa)
            
            # check for over or underflow
            if any(current > 1e10 for current in [INa, IK, Ileak, Iampa]):
                raise OverflowError("overflowed")
            elif any(abs(current) < 1e-10 for current in [INa, IK, Ileak, Iampa]):
                raise OverflowError("underflowed")
            
        except OverflowError as m:
            print(f"error: {m}")
            break

        na_currents.append(INa)
        k_currents.append(IK)
        leak_currents.append(Ileak)

        if include_ampa:
            ampa_currents.append(Iampa)

        
        # print(f"time is {i}")
        print(f"INa is {INa}")
        print(f"IK is {IK}")
        print(f"Ileak is {Ileak}")
        if include_ampa:
            print(f"Iampa is {Iampa}")
        
        print("\n")
        
        # sum the currents
        if include_ampa:
            Isum = stim[i] - INa - IK - Ileak -Iampa
        else:
            Isum = stim[i] - INa - IK - Ileak 

        Vm += deltaTms * Isum / Cm

        voltages[i] = Vm

        # update the voltages for each channel
        sodium_channel.Vm = Vm
        potassium_channel.Vm = Vm
        leaky_channel.Vm = Vm

        if include_ampa:
            ampa_receptor.Vm = Vm

            # when post synaptic fires weight updates
            if i >= 2:
                if voltages[i-2] <= voltages[i-1] and voltages[i-1] >= voltages[i]:
                    ampa_receptor.update_w(i)
                    print(f"this is activated at {i}")






