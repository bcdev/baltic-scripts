#!/usr/bin/env python
# encoding: utf-8
"""
Reader for OLCI ADFs
"""
import netCDF4
import numpy as np
# Local import
import get_bands
# Temporary function to spectrally adjust OLCI LUT to MSI
def adjust_S2_luts(adf_acp, adf_ppp):

    bands_sat_olci, *dummy = get_bands.main('OLCI', 'dummy')
    bands_sat_msi, *dummy = get_bands.main('S2MSI', 'dummy')
    bands_sat_olci, *dummy = get_bands.main('OLCI')
    bands_sat_msi, *dummy = get_bands.main('S2MSI')

    nband_olci = len(bands_sat_olci)
    nband_msi = len(bands_sat_msi)
    # Modify adf_acp
    adf_acp.bands = bands_sat_msi
    adf_acp.vicarious_gain = dict(zip(bands_sat_msi, np.ones(nband_msi)))
    # TODO adf_acp.rho_r but not used so far
    #  Modify adf_ppp CARE no extrapolation after OLCI 1020
    adf_ppp.bands = bands_sat_msi
    adf_ppp.tau_ray = np.exp(np.interp(np.log(bands_sat_msi), np.log(bands_sat_olci), np.log(adf_ppp.tau_ray))) # log scale interp
    adf_ppp.tau_no2_norm = np.interp(bands_sat_msi, bands_sat_olci, adf_ppp.tau_no2_norm)
    adf_ppp.tau_o3_norm = np.interp(bands_sat_msi, bands_sat_olci, adf_ppp.tau_o3_norm)
