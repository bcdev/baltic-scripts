#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os


def factor_in_ROT(altitude, pressure, latitude, bandshape=None, plotThis=False):
    # constants
    AVO = 6.0221367E+23  # Avogadro's number
    m_a_zero = 28.9595  # Mean molecular weight of dry air (zero H2O)
    g0_45 = 980.616  # Acceleration of gravity (sea level and 458 latitude)
    Ns = 2.5469E19  # Molecular density of gas in molecules / cm3 (288.15K, 1013.25mb)

    # Ns = Ns * 288.15/243.15 #-30°C in greenland in that summer..

    # constants describing the state of the atmosphere and which we don't know; better values may be used if known
    CO2 = 3.6E-4  # CO2 concentration at pixel; typical values are 300 to 360 ppm; here: parts per volumn
    C_CO2 = 360. / 10000.  # CO2 concentration in part per volume per percent
    m_a = 15.0556 * CO2 + m_a_zero  # mean molecular weight of dry air as function of actual CO2

    # Calculation to get the pressure
    ID = np.array(altitude < 0)
    altitude[ID] = 0.  # clip to sea level,

    # air pressure at the pixel (i.e. at altitude) in hPa, using the international pressure equation; T0=288.15K is correct!
    Psurf = pressure * (1. - 0.0065 * altitude / 288.15) ** 5.255
    P = Psurf * 1000.  # air pressure at pixel location in dyn / cm2, which is hPa * 1000

    # calculation to get the constant of gravity at the pixel altitude, taking the air mass above into account
    dphi = np.deg2rad(latitude)  # latitude in radians
    cos2phi = np.cos(2. * dphi)
    g0 = g0_45 * (1 - 0.0026373 * cos2phi + 0.0000059 * cos2phi ** 2)
    zs = 0.73737 * altitude + 5517.56  # effective mass-weighted altitude
    g = g0 - (3.085462E-4 + 2.27E-7 * cos2phi) * zs + (7.254E-11 + 1E-13 * cos2phi) * zs ** 2 - (
            1.517E-17 + 6E-20 * cos2phi) * zs ** 3
    # if plotThis:
    #     plt.imshow(altitude.reshape(bandshape), vmin=0., vmax=70.)
    #     plt.colorbar()
    #     plt.title('altitude')
    #     plt.show()
    #
    #     X = (1. - 0.0065 * altitude / 288.15) ** 5.255
    #     plt.imshow(X.reshape(bandshape), vmin=0.98, vmax=1.)
    #     plt.colorbar()
    #     plt.title('pressure conversion')
    #     plt.show()
    #
    #     plt.imshow(pressure.reshape(bandshape))
    #     plt.colorbar()
    #     plt.title('pressure')
    #     plt.show()
    #
    #     plt.imshow(P.reshape(bandshape))
    #     plt.title('air pressure at height')
    #     plt.colorbar()
    #     plt.show()
    #
    #     plt.imshow(g.reshape(bandshape))
    #     plt.title('gravity')
    #     plt.colorbar()
    #     plt.show()

    # calculations to get the Rayleigh optical thickness
    factor = (P * AVO) / (m_a * g)
    return factor


def rod_SRF(bandNo, co2=360., latitude=None, altitude=None, pressure=None, file_SRF_wavelength='', file_SRF_weights=''):
    factor = factor_in_ROT(altitude, pressure, latitude)

    SRF_lam = pd.read_csv(file_SRF_wavelength, sep='\t', header=None)
    SRF_weight = pd.read_csv(file_SRF_weights, sep='\t', header=None)

    ## wavelength dependent: scattering cross-section sigma!
    # constants describing the state of the atmosphere and which we don't know; better values may be used if known
    CO2 = co2 * 1E-6  # 3.6E-4  # CO2 concentration at pixel; typical values are 300 to 360 ppm; here: parts per volumn
    C_CO2 = co2 / 10000.  # CO2 concentration in part per volume per percent
    Ns = 2.5469E19  # Molecular density of gas in molecules / cm3 (288.15K, 1013.25mb)

    #print("[Rayleigh optical thickness with SRF]")
    #print("SRF: ", bandNo)
    lam_nm = SRF_lam.iloc[:, bandNo]  # wavelengths of band i in nm for spectral-response-function.

    lam = lam_nm / 1000.0  # wavelength in micrometer
    lam2 = lam_nm / 1.E7  # wavelength in cm
    F_N2 = 1.034 + 0.000317 / (lam ** 2)  # King factor of N2
    F_O2 = 1.096 + 0.001385 / (lam ** 2) + 0.0001448 / (lam ** 4)  # King factor of O2
    F_air = (78.084 * F_N2 + 20.946 * F_O2 + 0.934 * 1 + C_CO2 * 1.15) / (
            78.084 + 20.946 + 0.934 + C_CO2)  # depolarization ratio or King Factor, (6+3rho)/(6-7rho)
    n_ratio = 1 + 0.54 * (CO2 - 0.0003)
    # refractive index of dry air with 300ppm CO2:
    n_1_300 = (8060.51 + (2480990. / (132.274 - lam ** (-2))) + (17455.7 / (39.32957 - lam ** (-2)))) / 1.E8
    ## old: nCO2 = n_ratio * (1 + n_1_300)  # reflective index at CO2; THIS WAS AN ERROR!
    nCO2 = n_ratio * n_1_300 + 1  # reflective index at CO2
    sigma = (24 * np.pi ** 3 * (nCO2 ** 2 - 1) ** 2) / (lam2 ** 4 * Ns ** 2 * (nCO2 ** 2 + 2) ** 2) * F_air

    # plt.plot(lam_nm, SRF_weight.iloc[:, i], '-')
    # plt.plot(lam_nm, sigma/np.max(sigma), '-')
    # plt.show()
    lamSRF = np.sum(lam_nm * SRF_weight.iloc[:, bandNo]) / np.sum(SRF_weight.iloc[:, bandNo])
    sigma = np.sum(sigma * SRF_weight.iloc[:, bandNo]) / np.sum(SRF_weight.iloc[:, bandNo])
    # print(np.sum(SRF_weight.iloc[:, i]))

    taur_SRF = sigma * factor

    return taur_SRF, lamSRF
