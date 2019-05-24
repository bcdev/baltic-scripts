#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from misc import nlinear
from bodhaine import rod

def gas_correction(rho_toa, valid, latitude, longitude, yday, sza, vza, raa, wavelength, pressure, ozone, tcwv, adf_ppp, adf_clp, sensor):
    """
    Correct for gaseous absorption: O3, H2O, O2 and NO2
    Reference: OLCI ATBD  Correction of the impact of the absorption of atmospheric gases.
               Issue 2.2, 04/08/2010, ref. S3-L2-SD-03-C03-FUB-ATBD_GaseousCorrection
    H2O correction is based on the alternative FUB formulation using ECMWF WV (early tests for MER4RP/OLCI)
    """
    #TODO: handle S2

    # Initiliase rho_ng
    rho_ng = np.copy(rho_toa)

    # Compute air mass
    air_mass = 1./np.cos(np.radians(sza))+1./np.cos(np.radians(vza))

    # Correct for ozone all bands
    tO3 = np.exp(-adf_ppp.tau_o3_norm*ozone[valid,None]*air_mass[valid,None]) # pix*lambda product
    rho_ng[valid] /= tO3

    # Correct for O2 at 779 nm only
    i779 = list(adf_ppp.bands).index(779)
    U_O2 = pressure[valid] / adf_ppp.p_ref_t_o2
    wav = wavelength[valid,i779]
    LN = rho_toa[valid, i779]*np.cos(np.radians(sza[valid]))/np.pi # Ltoa/F0
    x = [U_O2, wav, LN, sza[valid], vza[valid], 180.-raa[valid]]
    axes = [adf_ppp.u_t_o2, adf_ppp.lambda_t_o2, adf_ppp.LN_t_o2, adf_ppp.SZA_t_o2, adf_ppp.VZA_t_o2, adf_ppp.RAA_t_o2]
    tO2 = nlinear(x, adf_ppp.t_o2_LUT, axes)
    rho_ng[valid, i779] /= tO2

    # Correct for H2O all bands but 709 nm. Note: according to S3 MPC this doesn't work ...
    UM = tcwv[valid]*air_mass[valid]
    for i, b in enumerate(adf_ppp.bands):
        if b == 709: continue
        tH2O = 0.
        for iw in range(adf_ppp.h2o_abs_bins[b]):
            tH2O += adf_ppp.h2o_relative_weights[b][iw]*np.exp(-adf_ppp.tau_h2o_norm[b][iw]*UM)
        if adf_ppp.h2o_abs_bins[b] > 0: rho_ng[valid, i] /= tH2O

    # Correct for H2O at 709 nm
    i = list(adf_ppp.bands).index(709)
    wav = wavelength[valid,i]
    nlambda_h2o_709 = len(adf_ppp.lambda_h2o_709)
    i1 = np.zeros(wav.shape,dtype='uint')
    i2 = np.zeros(wav.shape,dtype='uint') + nlambda_h2o_709 -1
    for l in range(nlambda_h2o_709):
        i1[wav >= adf_ppp.lambda_h2o_709[l]] = l
    for l in range(nlambda_h2o_709-1,-1,-1):
        i2[wav <= adf_ppp.lambda_h2o_709[l]] = l
    p709 = np.zeros(wav.shape,dtype='float32')
    idiff = i1 != i2
    if np.any(idiff):
        p709[idiff] = (wav[idiff] - adf_ppp.lambda_h2o_709[i1][idiff]) \
        / (adf_ppp.lambda_h2o_709[i2][idiff] - adf_ppp.lambda_h2o_709[i1][idiff])
    tH2O_1 = 0.
    tH2O_2 = 0.
    for iw in range(adf_ppp.h2o_max_bins_709):
        tH2O_1 += adf_ppp.h2o_709_relative_weights[i1,iw]*np.exp(-adf_ppp.tau_h2o_709_norm[i1,iw]*UM)
        tH2O_2 += adf_ppp.h2o_709_relative_weights[i2,iw]*np.exp(-adf_ppp.tau_h2o_709_norm[i2,iw]*UM)
    tH2O = (1. - p709) * tH2O_1 + p709 * tH2O_2
    rho_ng[valid, i] /= tH2O

    # Correct for NO2 at all bands
    yday = yday[valid]
    lat = latitude[valid]
    lon = longitude[valid]
    x = [yday, lat, lon]
    axes = [adf_clp.months_no2_clim, adf_clp.lat_no2_clim, adf_clp.lon_no2_clim]
    U_NO2 = nlinear(x, adf_clp.no2_clim_LUT, axes)
    tNO2 = np.exp(-adf_ppp.tau_no2_norm*U_NO2[:,None]*air_mass[valid,None])
    rho_ng[valid] /= tNO2

    return rho_ng


def glint_correction(rho_ng, valid, sza, oza, saa, raa, pressure, wind_u, wind_v, windm, adf_ppp):
    """
    Sun glint correction by Cox & Munk
    Reference: OLCI ATBD Glint correction. Issue 2.0, 08/04/10, ref. S3-L2-SD-03-C09-ARG-ATBD
    """

    # Initialise rho_g and signal corrected for sun glint
    rho_g = np.zeros(rho_ng.shape[0]) + np.NaN
    rho_gc = np.zeros(rho_ng.shape) + np.NaN

    # Compute wind azimuth in topocentric frame
    phiw = 180. - np.sign(wind_u[valid])*90. # default value for wind_v = 0
    index = wind_v[valid] > 0
    phiw[index] = np.degrees(np.arctan(wind_u[valid][index]/wind_v[valid][index]))
    index = wind_v[valid] < 0
    phiw[index] = 180. + np.degrees(np.arctan(wind_u[valid][index]/wind_v[valid][index]))

    # Compute wind azimuth in local frame
    wind_az = np.degrees(np.arccos(np.cos(np.radians(saa[valid] - phiw))))

    # Compute rho_g
    x = [wind_az, oza[valid], raa[valid], windm[valid], sza[valid]]
    axes = [adf_ppp.wind_az_rho_g, adf_ppp.VZA_rho_g, adf_ppp.RAA_rho_g, adf_ppp.windm_rho_g,adf_ppp.SZA_rho_g]
    rho_g[valid] = nlinear(x, adf_ppp.rho_g_LUT, axes)

    # Compute air mass and tdir
    air_mass = 1./np.cos(np.radians(sza))+1./np.cos(np.radians(oza))
    factor = pressure/adf_ppp.std_press*air_mass
    Tdir = np.exp(-adf_ppp.tau_ray*factor[:,None])

    # Correct sung glint at all bands
    rho_gc[valid] = rho_ng[valid] - Tdir[valid]*rho_g[valid,None]

    return rho_g, rho_gc

def white_caps_correction(rho_gc, valid, windm, td, adf_ppp):
    """
    White caps correction
    Reference: OLCI ATBD White Caps correction. Issue 2.0, 09/04/10, ref. S3-L2-SD-03-C06-ARG-ATBD
    """
    whitecaps_threshold = 5. # TODO read values in ADF
    whitecaps_alpha = 4.18E-5
    whitecaps_beta = 4.93

    # Initialise signal corrected for white caps
    rho_gwc = np.copy(rho_gc)

    # Do correction
    do_corr = valid & (windm > whitecaps_threshold)
    if np.any(do_corr):
        windm[windm>12] = 12.
        rho_wc = whitecaps_alpha*np.power(windm-whitecaps_beta,3.)
        rho_gwc[do_corr] -= td[do_corr]*rho_wc[do_corr,None]

    return rho_wc, rho_gwc

def Rayleigh_correction(rho_gc, valid, sza, oza, raa, pressure, windm, adf_acp, adf_ppp, sensor):
    """
    Rayleigh LUT of OLCI 
    """
    # TODO use instead Rayleigh LUT indexed by tau_r to handle exact wavelength and pressure correction
    # TODO handle S2

    # Initialise Rayleigh and Rayleigh corrected reflectance
    rho_r = np.zeros(rho_gc.shape)
    rho_rc = np.copy(rho_gc)

    # Interpolate Rayleigh at std_press
    x = [windm, sza, oza, raa]
    axes = [adf_acp.wind, adf_acp.sza, adf_acp.vza, adf_acp.raa]
    ipress = np.argmin(abs(adf_acp.pressure-adf_ppp.std_press)) # TODO normaly ipress should be pixel dependent
    for i, b in enumerate(adf_acp.bands):
        rho_r[:,i] = nlinear(x, adf_acp.rho_r[ipress,i,:,:,:,:], axes)

    # Correct for pressure around std_press
    factor = pressure/adf_ppp.std_press
    rho_r *= factor[:,None]

    rho_rc[valid] -= rho_r[valid]

    return rho_r, rho_rc

def diffuse_transmittance(sza, oza, pressure, adf_ppp):
    """
    Total direct+diffuse transmittance (upward+backward) for Rayleigh
    """
    #TODO Take LUT instead of analytical formula

    # Compute air mass
    air_mass = 1./np.cos(np.radians(sza))+1./np.cos(np.radians(oza))

    # Compute td
    td = np.exp(-0.5*adf_ppp.tau_ray*pressure[:,None]/adf_ppp.std_press*air_mass[:,None])
    td = np.array(td, dtype='float64')

    return td

def Rmolgli_correction_Hygeos(rho_ng, valid, latitude, sza, oza, raa, wavelength, pressure, windm, LUT):
    """
    Rayleigh + glint correction from HYGEOS LUT
    This includes correction for pressure and smile
    Care : rho_r is also return without the glint term
    """
    # Initialise Rayleigh and Rayleigh corrected reflectance
    rho_molgli = np.zeros(rho_ng.shape) + np.NaN
    rho_rc = np.copy(rho_ng)
    tau_ray = np.zeros(rho_ng.shape)

    # Compute Rayleigh optical thickness from Bodhaine
    co2 = 400.
    altitude = 0.
    for i in range(tau_ray.shape[1]):
        tau_ray[:,i] = rod(wavelength[:,i]/1000., co2, latitude, altitude, pressure)

    # Compute rho_molgli (dim_mu, dim_phi, dim_mu, dim_tauray, dim_wind)
    axes = [LUT.muv, LUT.raa, LUT.mus, LUT.tau, LUT.wind]
    for i in range(tau_ray.shape[1]):
        x = [np.cos(np.radians(oza)), raa, np.cos(np.radians(sza)), tau_ray[:,i], windm]
        rho = nlinear(x, LUT.rho_molgli, axes) # Rayleigh+glint
        rho_molgli[valid,i] = rho[valid]

    # Correct for Rayleigh + glint
    rho_rc -= rho_molgli

    # Compute rho_r only for the AC inversion
    rho_r = np.zeros(rho_ng.shape)
    axes = [LUT.muv, LUT.raa, LUT.mus, LUT.tau]
    for i in range(tau_ray.shape[1]):
        x = [np.cos(np.radians(oza)), raa, np.cos(np.radians(sza)), tau_ray[:,i]]
        rho_r[:,i] = nlinear(x, LUT.rho_mol, axes) # Rayleigh

    return rho_r, rho_molgli, rho_rc

