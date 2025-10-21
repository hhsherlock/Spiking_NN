#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13

@author: yaning
"""

import pickle
import torch
import math

class Params:
    def __init__(self):
        self.u_se_ampa = 0.5
        self.u_se_nmda = 0.5
        self.u_se_gaba = 0.5
        self.tau_rec_ampa = 5.0
        self.tau_rec_nmda = 12.0
        self.tau_rec_gaba = 12.0
        self.tau_rise_ampa = 15.0
        self.tau_rise_nmda = 150.0
        self.tau_rise_gaba = 15.0
        self.learning_rate = 1.0
        self.weight_scale = 1.0

params = Params()

