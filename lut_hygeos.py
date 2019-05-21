#!/usr/bin/env python
# encoding: utf-8

""" 
Reader for HYGEOS LUT
"""

from pyhdf.SD import SD
import numpy as np

class LUT(object):
    '''
    Read HYGEOS LUT
    '''
    def __init__(self, adffile):

        hdf = SD(adffile)

        # Read mu angles
        sds = hdf.select('dim_mu')
        self.muv = sds.get()[::-1] # put in ascending order for mu
        self.mus = np.copy(self.muv)

        # Read raa
        sds = hdf.select('dim_phi')
        self.raa = sds.get()

        # Read tau_ray
        sds = hdf.select('dim_tauray')
        self.tau = sds.get()

        # Read wind
        sds = hdf.select('dim_wind')
        self.wind = sds.get()

        # Read Rmolgli
        sds = hdf.select('Rmolgli')
        rho = sds.get() # (mu, raa, mu, tau, wind)
        self.rho_molgli = rho#np.flip(rho,(0,2)) # put in ascending order for mu

        # Read Rmol
        sds = hdf.select('Rmol')
        rho = sds.get() # (mu, raa, mu, tau)
        self.rho_mol = rho#np.flip(rho,(0,2)) # put in ascending order for mu
 

