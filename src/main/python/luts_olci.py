#!/usr/bin/env python
# encoding: utf-8

""" 
Reader for OLCI ADFs
"""

import netCDF4
import numpy as np


class LUT_ACP(object):
    '''
    Read ACP LUT
    '''
    def __init__(self, adffile):
        bands_sat  = np.array([400,412,443,490,510,560,
                            620,665,674,681,709,754,
                            760,764,767,779,865,885,
                            900,940,1020], dtype=int) # TODO bands_sat should be define once for all in a unique global vector
        nband_sat = len(bands_sat)
        self.bands = bands_sat

        grpname = 'glint_whitecaps'
        nci = netCDF4.Dataset(adffile, 'r')
        grp = nci.groups[grpname]
        vicarious_gain = {}
        for i, b in enumerate(bands_sat):
            vicarious_gain[b] = grp.variables['gain_vicarious'][i]
        self.vicarious_gain = vicarious_gain

        grpname = 'standard_AC'
        grp = nci.groups[grpname]
        rho_r = grp.variables['rho_rayleigh_LUT'][:] # ref_press, bands, wind_speeds, SZA, VZA, RAA
        self.rho_r = rho_r
        self.pressure = grp.variables['ref_press'][:]
        self.wind = grp.variables['wind_speeds'][:]
        self.sza = grp.variables['SZA'][:]
        self.vza = grp.variables['VZA'][:]
        self.raa = grp.variables['RAA'][:]
        nci.close()


class LUT_PPP(object):
    '''
    Read PPP LUT
    '''
    def __init__(self, adffile):
        nci = netCDF4.Dataset(adffile, 'r')
        
        bands_sat  = np.array([400,412,443,490,510,560,
                            620,665,674,681,709,754,
                            760,764,767,779,865,885,
                            900,940,1020], dtype=int) # TODO bands_sat should be define once for all in a unique global vector
        nband_sat = len(bands_sat)
        self.bands = bands_sat

        # Rayleigh optical thickness and standard pressure
        self.tau_ray = nci.variables['tau_rayleigh'][:]
        self.std_press = nci.variables['standard_pressure'][:]

        # Group for gaseous correction
        grpname = 'gas_correction'
        grp = nci.groups[grpname]

        # NO2 parameters
        self.tau_no2_norm = grp.variables['tau_no2_norm'][:]

        # O3 parameters
        self.tau_o3_norm = grp.variables['tau_o3_norm'][:]

        # O2 parameters
        self.p_ref_t_o2 = grp.variables['p_ref_t_o2'][:]
        self.u_t_o2 = grp.variables['u_t_o2'][:]
        self.lambda_t_o2 = grp.variables['lambda_t_o2'][:]
        self.LN_t_o2 = grp.variables['LN_t_o2'][:]
        self.SZA_t_o2 = grp.variables['SZA_t_o2'][:]
        self.VZA_t_o2 = grp.variables['VZA_t_o2'][:]
        self.RAA_t_o2 = grp.variables['RAA_t_o2'][:]
        self.t_o2_LUT = grp.variables['t_o2_LUT'][:] # shape is (lambda_t_o2, LN_t_o2, SZA_t_o2, VZA_t_o2, RAA_t_o2) 

        # H2O parameters #
        self.h2o_max_bins = grp.dimensions['h2o_max_bins'].size
        self.lambda_h2o = grp.variables['lambda_h2o'][:]
        self.h2o_abs_bins = {}
        self.tau_h2o_norm = {}
        self.h2o_relative_weights = {}
        for i, b in enumerate(self.lambda_h2o):
            self.h2o_abs_bins[b] = grp.variables['h2o_abs_bins'][i]
            self.tau_h2o_norm[b] = grp.variables['tau_h2o_norm'][i]
            self.h2o_relative_weights[b] = grp.variables['h2o_relative_weights'][i]

        self.h2o_max_bins_709 = grp.dimensions['h2o_max_bins_709'].size
        self.h2o_709_abs_bins = grp.variables['h2o_709_abs_bins'][:]
        self.lambda_h2o_709 = grp.variables['lambda_h2o_709'][:]
        self.tau_h2o_709_norm = grp.variables['tau_h2o_709_norm'][:]
        self.h2o_709_relative_weights = grp.variables['h2o_709_relative_weights'][:]

        # Glint LUT
        grpname = 'classification_1'
        grp = nci.groups[grpname]
        self.wind_az_rho_g = grp.variables['wind_az_rho_g'][:]
        self.VZA_rho_g = grp.variables['VZA_rho_g'][:]
        self.RAA_rho_g = grp.variables['RAA_rho_g'][:]
        self.windm_rho_g = grp.variables['windm_rho_g'][:]
        self.SZA_rho_g = grp.variables['SZA_rho_g'][:]
        self.rho_g_LUT = grp.variables['rho_g_LUT'][:] #(wind_az_rho_g, VZA_rho_g, RAA_rho_g, windm_rho_g, SZA_rho_g)

        nci.close()
 
class LUT_CLP():
    '''
    Read CLP LUT
    '''
    def __init__(self, adffile):
        nci = netCDF4.Dataset(adffile, 'r')

        # NO2 climatology
        months_no2_clim = nci.variables['months_no2_clim'][:]
        lat_no2_clim = np.flipud(nci.variables['lat_no2_clim'][:]) # flip to put lat in ascending order - but don't do it for no2_clim_LUT (ADF bug?)
        lon_no2_clim = nci.variables['lon_no2_clim'][:]
        no2_clim_LUT = nci.variables['no2_clim_LUT'][:] # shape is (months_no2_clim, lat_no2_clim, lon_no2_clim)

        # Extend NO2 LUT to make it cyclic versus month
        no2_clim_LUT_cycl = np.ndarray((14,len(lat_no2_clim),len(lon_no2_clim)))
        no2_clim_LUT_cycl[1:13,:,:] = no2_clim_LUT[:,:,:]
        no2_clim_LUT_cycl[0,:,:] = no2_clim_LUT[-1,:,:]
        no2_clim_LUT_cycl[13,:,:] = no2_clim_LUT[0,:,:]
        no2_clim_LUT = no2_clim_LUT_cycl
        months_no2_clim_cycl = np.ndarray(14)
        months_no2_clim_cycl[1:13] = months_no2_clim
        months_no2_clim_cycl[0] = months_no2_clim[-1] - 365.
        months_no2_clim_cycl[13] = 365. + months_no2_clim[0]
        months_no2_clim = months_no2_clim_cycl

        nci.close()

        self.months_no2_clim = months_no2_clim
        self.lat_no2_clim = lat_no2_clim
        self.lon_no2_clim = lon_no2_clim
        self.no2_clim_LUT = no2_clim_LUT

