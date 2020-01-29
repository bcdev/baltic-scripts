#!/usr/bin/python
# -*- coding: utf-8 -*-
import collections
import datetime
import json
#from keras.models import load_model
import locale
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
import sys
import time
import glob

# snappy import
sys.path.append('/home/cmazeran/.snap/snap-python')
#sys.path.append("C:\\Users\Dagmar\.snap\snap-python")
import snappy as snp
from snappy import Product
from snappy import ProductData
from snappy import ProductDataUTC
from snappy import ProductIO
from snappy import ProductUtils
from snappy import ProgressMonitor
from snappy import FlagCoding
from snappy import jpy
from snappy import GPF
from snappy import HashMap
#from snappy import TimeCoding #org.esa.snap.core.datamodel.TimeCoding
from snappy import PixelPos #org.esa.snap.core.datamodel.PixelPos

#fetchOzone = jpy.get_type('org.esa.s3tbx.c2rcc.ancillary.AncillaryCommons.fetchOzone')
AtmosphericAuxdata = jpy.get_type('org.esa.s3tbx.c2rcc.ancillary.AtmosphericAuxdata')
AtmosphericAuxdataBuilder = jpy.get_type('org.esa.s3tbx.c2rcc.ancillary.AtmosphericAuxdataBuilder')
TimeCoding = jpy.get_type('org.esa.snap.core.datamodel.TimeCoding')
AncDownloader = jpy.get_type('org.esa.s3tbx.c2rcc.ancillary.AncDownloader')
AncillaryCommons = jpy.get_type('org.esa.s3tbx.c2rcc.ancillary.AncillaryCommons')
AncRepository = jpy.get_type('org.esa.s3tbx.c2rcc.ancillary.AncRepository')
File = jpy.get_type('java.io.File')

AncDataFormat = jpy.get_type('org.esa.s3tbx.c2rcc.ancillary.AncDataFormat')
Calendar = jpy.get_type('java.util.Calendar')


# local import
from baltic_corrections import gas_correction, glint_correction, white_caps_correction, Rayleigh_correction, diffuse_transmittance, Rmolgli_correction_Hygeos, vicarious_calibration
import get_bands
from misc import default_ADF, nlinear
import luts_olci
import lut_hygeos
from auxdata_handling import setAuxData, checkAuxDataAvailablity, getGeoPositionsForS2Product, yearAndDoyAndHourUTC

# Set locale for proper time reading with datetime
locale.setlocale(locale.LC_ALL, 'en_US.UTF_8')

# Define NNs as global variables
def read_NNs(sensor, NNversion):
    global nnForwardFilePath, nnBackwardFilePath, nnNormFilePath
    global nn_iop_rw, nn_rw_iop, nn_rw_rwnorm

    # Define paths
    if sensor == 'OLCI':
        if NNversion == 'v2':
            nnForwardFilePath = "forwardNN_c2rcc/olci/olci_20171221/iop_rw/77x77x77_1798.8.net"
            nnBackwardFilePath = "forwardNN_c2rcc/olci/olci_20171221/rw_iop/37x37x37_596495.4.net"
            nnNormFilePath = "forwardNN_c2rcc/olci/olci_20171221/rw_rwnorm/77x77x77_34029.1.net"
        elif NNversion == 'v3':
            nnForwardFilePath = "forwardNN_c2rcc/olci/olci_20190414/iop_rw/55x55x55_40.3.net"
            nnBackwardFilePath = "forwardNN_c2rcc/olci/olci_20190414/rw_iop/55x55x55_99.6.net"
            nnNormFilePath = "forwardNN_c2rcc/olci/olci_20190414/rw_rwnorm/77x77x77_34029.1.net"
    elif sensor == 'S2MSI':
        nnForwardFilePath = "forwardNN_c2rcc/msi/std_s2_20160502/iop_rw/17x97x47_125.5.net"
        nnBackwardFilePath = "forwardNN_c2rcc/msi/std_s2_20160502/rw_rwnorm/27x7x27_28.0.net"
        nnNormFilePath = "forwardNN_c2rcc/msi/std_s2_20160502/rw_rwnorm/27x7x27_28.0.net"

    # Read the NNs
    NNffbpAlphaTabFast = jpy.get_type('org.esa.snap.core.nn.NNffbpAlphaTabFast')
    nnfile = open(nnForwardFilePath, 'r')
    nnCode = nnfile.read()
    nn_iop_rw = NNffbpAlphaTabFast(nnCode)
    nnfile = open(nnBackwardFilePath, 'r')
    nnCode = nnfile.read()
    nn_rw_iop = NNffbpAlphaTabFast(nnCode)
    nnfile = open(nnNormFilePath, 'r')
    nnCode = nnfile.read()
    nn_rw_rwnorm = NNffbpAlphaTabFast(nnCode)

def read_NN_input_ranges_fromFile(nnFilePath):
    """ Read input range for the forward NN """
    input_range = collections.OrderedDict()
    with open(nnFilePath) as f:
        lines = f.readlines()
        for l in lines:
            if 'input' in l[:5]:
                #input  1 is sun_zenith in [0.000010,75.000000]
                v = [a for a in l.split(' ') if a!='']
                varname = v[3]
                r = v[5].split(',')
                r = (float(r[0][1:]), float(r[1][:-2]))
                input_range[varname] = r
    f.close()
    return input_range

def get_band_or_tiePointGrid(product, name, dtype='float32', reshape=True):
    ##
    # This function reads a band or tie-points, identified by its name <name>, from SNAP product <product>
    # The fuction returns a numpy array of shape (height, width)
    ##
    height = product.getSceneRasterHeight()
    width = product.getSceneRasterWidth()
    var = np.zeros(width * height, dtype=dtype)
    if name in list(product.getBandNames()):
        product.getBand(name).readPixels(0, 0, width, height, var)
    elif name in list(product.getTiePointGridNames()):
        var.shape = (height, width)
        for i in range(height):
            for j in range(width):
                var[i, j] = product.getTiePointGrid(name).getPixelDouble(j, i)
        var.shape = (height*width)
    else:
        raise Exception('{}: neither a band nor a tie point grid'.format(name))

    if reshape:
        var.shape = (height, width)

    return var

def Level1_Reader(product, sensor, band_group='radiance', reshape=True):
    input_label = []
    if sensor == 'S2MSI':
        input_label = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
    elif sensor == 'OLCI':
        if band_group == 'radiance':
            input_label = ["Oa%02d_radiance"%(i+1) for i in range(21)]
        elif band_group == 'solar_flux':
            input_label = ["solar_flux_band_%d"%(i+1) for i in range(21)]
        elif band_group == 'lambda0':
            input_label = ["lambda0_band_%d"%(i+1) for i in range(21)]

    # Initialise and read all bands contained in input_label (pixel x band)
    height = product.getSceneRasterHeight()
    width = product.getSceneRasterWidth()
    var = np.zeros((width * height, len(input_label)))
    for i, bn in enumerate(input_label):
        var[:,i] = get_band_or_tiePointGrid(product, bn, reshape=reshape)

    return var

def get_yday(product,reshape=True):
    # Get product size
    height = product.getSceneRasterHeight()
    width = product.getSceneRasterWidth()

    ## time handling for match-up files
    if str(product.getFileLocation()).endswith('.txt') or str(product.getFileLocation()).endswith('.csv'):
        yday = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                a = product.getSceneTimeCoding().getMJD(PixelPos(x + 0.5, y + 0.5))
                year, yday[y, x], hour = yearAndDoyAndHourUTC(a)

    else:
        ## time handling for a scene
        # Get yday of each row
        dstart = datetime.datetime.strptime(str(product.getStartTime()),'%d-%b-%Y %H:%M:%S.%f')
        dstop = datetime.datetime.strptime(str(product.getEndTime()),'%d-%b-%Y %H:%M:%S.%f')
        dstart = np.datetime64(dstart)
        dstop = np.datetime64(dstop)
        dpix = dstart + (dstop-dstart)/float(height-1.)*np.arange(height)
        yday = [k.timetuple().tm_yday for k in dpix.astype(datetime.datetime)]

        # Apply to all columns
        yday = np.array(yday*width).reshape(width,height).transpose()

    if not reshape:
        yday = np.ravel(yday)

    return yday

def radianceToReflectance_Reader(product, sensor = '', print_info=False):

    if print_info:
        height = product.getSceneRasterHeight()
        width = product.getSceneRasterWidth()
        name = product.getName()
        description = product.getDescription()
        band_names = product.getBandNames()

        print("Sensor:      %s" % sensor)
        print("Product:     %s, %s" % (name, description))
        print("Raster size: %d x %d pixels" % (width, height))
        print("Start time:  " + str(product.getStartTime()))
        print("End time:    " + str(product.getEndTime()))
        print("Bands:       %s" % (list(band_names)))

    if sensor == 'OLCI':
        rad = Level1_Reader(product, sensor, band_group='radiance',reshape=False)
        solar_flux = Level1_Reader(product, sensor, band_group='solar_flux',reshape=False)
        SZA = get_band_or_tiePointGrid(product, 'SZA', reshape=False)
        refl = rad * np.pi / (solar_flux * np.cos(SZA.reshape(rad.shape[0],1)*np.pi/180.))
    elif sensor == 'S2MSI':
        refl = Level1_Reader(product, sensor,reshape=False)

    return refl

def angle_Reader(product, sensor):
    if sensor == 'OLCI':
        oaa = get_band_or_tiePointGrid(product, 'OAA', reshape=False)
        oza = get_band_or_tiePointGrid(product, 'OZA', reshape=False)
        saa = get_band_or_tiePointGrid(product, 'SAA', reshape=False)
        sza = get_band_or_tiePointGrid(product, 'SZA', reshape=False)
    elif sensor == 'S2MSI':
        oaa = get_band_or_tiePointGrid(product, 'view_azimuth_mean', reshape=False)
        oza = get_band_or_tiePointGrid(product, 'view_zenith_mean', reshape=False)
        saa = get_band_or_tiePointGrid(product, 'sun_azimuth', reshape=False)
        sza = get_band_or_tiePointGrid(product, 'sun_zenith', reshape=False)

    return oaa, oza, saa, sza

def calculate_diff_azim(oaa, saa):
    ###
    # MERIS/OLCI ground segment definition
    raa = np.degrees(np.arccos(np.cos(np.radians(oaa - saa))))
    ###
    # azimuth difference as input to the NN is defined in a range between 0° and 180°
    ## from c2rcc JAVA implementation
    nn_raa = np.abs(180 + oaa - saa)
    ID = np.array(nn_raa > 180)
    if np.sum(ID) > 0:
        nn_raa = 360 - nn_raa

    return raa, nn_raa

def run_IdePix_processor(product, sensor):
    ### todo:  check, if L1 flags are given as band 'quality_flags'
    # (in CalValus extracts this single band representation of the flags might be missing.)
    # calculate single band 'quality_flags' if necessary.
    # invoke IdePix.
    # define valid pixel expression.
    idepixParameters = HashMap()
    idepixParameters.put("computeCloudBuffer", 'true')
    idepixParameters.put("cloudBufferWidth", '2')

    idepix_product = None

    if sensor == 'OLCI':
        idepix_product = GPF.createProduct("Idepix.Olci", idepixParameters, product) # SNAP v7
    elif sensor == 'S2MSI':
        idepixParameters.put("computeCloudBufferForCloudAmbiguous", 'true')
        idepix_product = GPF.createProduct("Idepix.S2", idepixParameters, product) # SNAP v7

    return idepix_product

def check_valid_pixel_expression_Idepix(product, sensor):
    valid_pixel_flag = None
    if sensor == 'OLCI':
        height = product.getSceneRasterHeight()
        width = product.getSceneRasterWidth()
        quality_flags = np.zeros(width * height, dtype='uint32')
        product.getBand('pixel_classif_flags').readPixels(0, 0, width, height, quality_flags)

        # Masks OLCI Idepix
        invalid_mask = np.bitwise_and(quality_flags, 2 ** 0) == 2 ** 0
        cloud_mask = np.bitwise_and(quality_flags, 2 ** 1) == 2 ** 1
        cloudbuffer_mask = np.bitwise_and(quality_flags, 2 ** 4) == 2 ** 4
        snowice_mask = np.bitwise_and(quality_flags, 2 ** 6) == 2 ** 6

        invalid_mask = np.logical_or(np.logical_or(invalid_mask, snowice_mask), np.logical_or(cloud_mask, cloudbuffer_mask))
        valid_pixel_flag = np.logical_not(invalid_mask)

    if sensor == 'S2MSI':
        height = product.getSceneRasterHeight()
        width = product.getSceneRasterWidth()
        quality_flags = np.zeros(width * height, dtype='uint32')
        product.getBand('pixel_classif_flags').readPixels(0, 0, width, height, quality_flags)

        # Masks S2MSI Idepix
        invalid_mask = np.bitwise_and(quality_flags, 2 ** 0) == 2 ** 0
        cloud_mask = np.bitwise_and(quality_flags, 2 ** 1) == 2 ** 1
        cloudbuffer_mask = np.bitwise_and(quality_flags, 2 ** 4) == 2 ** 4
        snowice_mask = np.bitwise_and(quality_flags, 2 ** 6) == 2 ** 6

        invalid_mask = np.logical_or(np.logical_or(invalid_mask, snowice_mask),
                                     np.logical_or(cloud_mask, cloudbuffer_mask))
        valid_pixel_flag = np.logical_not(invalid_mask)

    return valid_pixel_flag


def check_valid_pixel_expression_L1(product, sensor):
    valid_pixel_flag = None
    if sensor == 'OLCI':
        height = product.getSceneRasterHeight()
        width = product.getSceneRasterWidth()
        quality_flags = np.zeros(width * height, dtype='uint32')

        # match-up extractions from Calvalus do not have a 'quality_flags' band
        if product.getBand('quality_flags') is None:
            invalid_mask = np.zeros(width * height, dtype='uint32')
            product.getBand('quality_flags.invalid').readPixels(0, 0, width, height, invalid_mask)
            land_mask = np.zeros(width * height, dtype='uint32')
            product.getBand('quality_flags.land').readPixels(0, 0, width, height, land_mask)
            coastline_mask =  np.zeros(width * height, dtype='uint32')
            product.getBand('quality_flags.coastline').readPixels(0, 0, width, height, coastline_mask)
            inland_mask =  np.zeros(width * height, dtype='uint32')
            product.getBand('quality_flags.fresh_inland_water').readPixels(0, 0, width, height, inland_mask)

            land_mask = coastline_mask | (land_mask & ~inland_mask)

            bright_mask = np.zeros(width * height, dtype='uint32')
            product.getBand('quality_flags.bright').readPixels(0, 0, width, height, bright_mask)

            invalid_mask = np.logical_or(invalid_mask, np.logical_or(land_mask, bright_mask))
            valid_pixel_flag = np.logical_not(invalid_mask)
        else:
            product.getBand('quality_flags').readPixels(0, 0, width, height, quality_flags)

            # Masks OLCI L1
            ## flags: 31=land 30=coastline 29=fresh_inland_water 28=tidal_region 27=bright 26=straylight_risk 25=invalid
            ## 24=cosmetic 23=duplicated 22=sun-glint_risk 21=dubious 20->00=saturated@Oa01->saturated@Oa21
            ## todo: could also include simple bright NIR test: rho_toa(865nm) < 0.2
            invalid_mask = np.bitwise_and(quality_flags, 2 ** 25) == 2 ** 25
            land_mask = np.bitwise_and(quality_flags, 2 ** 31) == 2 ** 31
            coastline_mask = np.bitwise_and(quality_flags, 2 ** 30) == 2 ** 30
            inland_mask = np.bitwise_and(quality_flags, 2 ** 29) == 2 ** 29
            land_mask = coastline_mask | (land_mask & ~inland_mask)

            bright_mask = np.bitwise_and(quality_flags, 2 ** 27) == 2 ** 27

            invalid_mask = np.logical_or(invalid_mask , np.logical_or( land_mask , bright_mask))
            valid_pixel_flag = np.logical_not(invalid_mask)

    elif sensor == 'S2MSI':
        #TODO: set valid pixel expression L1C S2
        height = product.getSceneRasterHeight()
        width = product.getSceneRasterWidth()
        valid_pixel_flag = np.ones(width * height, dtype='uint32')

        b8 = get_band_or_tiePointGrid(product, 'B8', reshape=False)
        valid_pixel_flag = np.logical_and(np.array(b8 > 0), np.array(b8 < 0.1))

    return valid_pixel_flag

def apply_forwardNN_IOP_to_rhow(log_iop, sun_zenith, view_zenith, diff_azimuth, sensor, valid_data, iband_NN_forward, T=15, S=35):
    """
    Apply the forwardNN: IOP to rhow
    input: numpy array log_iop, shape = (Npixels x log_iops= (log_apig, log_adet, log a_gelb, log_bpart, log_bwit)),
            np.array sza, shape = (Npixels,)
            np.array oza, shape = (Npixels,)
            np.array raa, shape = (Npixels,); range: 0-180
    returns: np.array rhow, shape = (Npixels, wavelengths)

    T, S: currently constant #TODO take ECMWF temperature at sea surface?
    valid ranges can be found at the beginning of the .net-file.

    NN output for OLCI (12 bands): log_rw at lambda = 400, 412, 443, 489, 510, 560, 620, 665, 674, 681, 709, 754
    NN output for S2MSI (6 bands): log_rw at lambda = 443, 490, 560, 665, 705, 740
    """

    # Initialise output
    nBands = len(iband_NN_forward)
    output = np.zeros((log_iop.shape[0], nBands)) + np.NaN

    ###
    # Launch the NN
    # Important:
    # OLCI input array has to be of size 10: [SZA, VZA, RAA, T, S, log_apig, log_adet, log a_gelb, log_bpart, log_bwit]
    # S2 input array has to be of size 10 (same order as OLCI): [ sun_zeni, view_zeni, azi_diff, T, S, log_conc_apig, log_conc_adet,
    # log_conc_agelb, log_conc_bpart, log_conc_bwit]
    inputNN = np.zeros(10) -1. # care zero used by default for de-activated IOPs when niop<5
    inputNN[3] = T
    inputNN[4] = S
    for i in range(log_iop.shape[0]):
        if valid_data[i]:
            inputNN[0] = sun_zenith[i]
            inputNN[1] = view_zenith[i]
            inputNN[2] = diff_azimuth[i]
            for j in range(log_iop.shape[1]): # log_apig, log_adet, log a_gelb, log_bpart, log_bwit
                inputNN[j+5] = log_iop[i, j]
            log_rw_nn2 = np.array(nn_iop_rw.calc(inputNN), dtype=np.float32)
            output[i, :] = np.exp(log_rw_nn2)


    # #// (9.5.4)
    # #check if log_IOPs out of range
    # mi = nn_rw_iop.getOutmin();
    # ma = nn_rw_iop.getOutmax();
    # boolean iop_oor_flag = false;
    # for (int iv = 0; iv < log_iops_nn1.length; iv++) {
    # 	if (log_iops_nn1[iv] < mi[iv] | log_iops_nn1[iv] > ma[iv]) {
    # 		iop_oor_flag = true;
    # 	}
    # }
    # flags = BitSetter.setFlag(flags, FLAG_INDEX_IOP_OOR, iop_oor_flag);
    #
    # #// (9.5.5)
    # # check if log_IOPs	at limit
    # int firstIopMaxFlagIndex = FLAG_INDEX_APIG_AT_MAX;
    # for (int i = 0; i < log_iops_nn1.length; i++) {
    # 	final boolean iopAtMax = log_iops_nn1[i] > (ma[i] - log_threshfak_oor);
    # 	flags = BitSetter.setFlag(flags, i + firstIopMaxFlagIndex, iopAtMax);
    # }
    #
    # int	firstIopMinFlagIndex = FLAG_INDEX_APIG_AT_MIN;
    # for (int i = 0; i < log_iops_nn1.length; i++) {
    # 	final boolean iopAtMin = log_iops_nn1[i] < (mi[i] + log_threshfak_oor);
    # 	flags = BitSetter.setFlag(flags, i + firstIopMinFlagIndex, iopAtMin);
    # }
    #
    #

    return output

def apply_backwardNN_rhow_to_IOP(rhow, sun_zenith, view_zenith, diff_azimuth, sensor, valid_data, nn_rw_iop, iband_NN_backward, inputRange_backward, T=15, S=35):
    """
    Apply the backwardNN: rhow to IOP
    input: numpy array rhow, shape = (Npixels x rhow = (log_rw_band1, log_rw_band2_.... )),
            np.array sza, shape = (Npixels,)
            np.array oza, shape = (Npixels,)
            np.array raa, shape = (Npixels,); range: 0-180
    returns: np.array log_iop, shape = (Npixels x log_iop= (log_apig, log_adet, log a_gelb, log_bpart, log_bwit))

    T, S: currently constant #TODO take ECMWF temperature at sea surface?
    valid ranges can be found at the beginning of the .net-file.

    NN input for OLCI (12 bands): log_rw at lambda = 400, 412, 443, 489, 510, 560, 620, 665, 674, 681, 709, 754
    NN input for S2MSI (8 bands): log_rw at lambda = 443, 490, 560, 665, 705, 740, 783, 865 
    """

    # Initialise output
    nBands = len(iband_NN_backward)
    niop = 5
    output = np.zeros((rhow.shape[0], niop)) + np.NaN
    FlagConstraintApplied = np.zeros(rhow.shape[0])

    ###
    # Launch the NN
    # CARE to the input bands
    inputNN = np.zeros(nBands+5) -1. # care zero used by default for de-activated IOPs when niop<5
    inputNN[3] = T
    inputNN[4] = S

    for i in range(rhow.shape[0]):
        if valid_data[i]:
            inputNN[0] = sun_zenith[i]
            inputNN[1] = view_zenith[i]
            inputNN[2] = diff_azimuth[i]
            for j,var in enumerate(list(inputRange_backward.keys())[5:]):
                # Threshold input rhow, in case of negative or too high value
                j_glob = iband_NN_backward[j]
                # inputRange=list(inputRange_backward.keys())[5:][j]
                rhow_in = max(rhow[i, j_glob], np.exp(inputRange_backward[var][0]))
                rhow_in = min(rhow_in, np.exp(inputRange_backward[var][1]))
                if rhow_in != rhow[i, j_glob]:
                    FlagConstraintApplied[i] = 1
                inputNN[j+5] = np.log(rhow_in)
            output[i,:] = np.array(nn_rw_iop.calc(inputNN), dtype=np.float32) # log_apig, log_adet, log a_gelb, log_bpart, log_bwit

    return output, FlagConstraintApplied

def apply_NN_rhow_to_rhownorm(rhow, sun_zenith, view_zenith, diff_azimuth, sensor, valid_data, nn_rw_rwnorm, iband_NN_norm, inputRange_norm, T=15, S=35):
    """
    Apply NN :  rhow to rhownorm
    input: numpy array rhow, shape = (Npixels x wavelengths),
            np.array sza, shape = (Npixels,)
            np.array oza, shape = (Npixels,)
            np.array raa, shape = (Npixels,); range: 0-180
    returns: np.array rhownorm, shape = (Npixels, wavelengths)

    NN input and output for OLCI (12 bands): log_rw_norm at lambda = 400, 412, 443, 489, 510, 560, 620, 665, 674, 681, 709, 754
    NN input and output for S2MSI (6 bands): log_rw_norm at lambda = 443, 490, 560, 665, 705, 740
    """

    # Initialise output
    nBands = None
    if sensor == 'OLCI':
        nBands = 12
    elif sensor == 'S2MSI':
        nBands = 6
    output = np.zeros((rhow.shape[0], nBands)) + np.NaN

    ###
    # Launch the NN
    # Important:
    # OLCI input array has to be of size 17: [SZA, VZA, RAA, T, S, 12x log rhow]
    # S2 input array has to be of size 11 : [ sun_zeni, view_zeni, azi_diff, T, S, 6x log rhow]
    inputNN = np.zeros(5+nBands)
    inputNN[3] = T
    inputNN[4] = S

    FlagConstraintApplied = np.zeros(rhow.shape[0])

    for i in range(rhow.shape[0]):
        if valid_data[i]:
            inputNN[0] = sun_zenith[i]
            inputNN[1] = view_zenith[i]
            inputNN[2] = diff_azimuth[i]
            for j,var in enumerate(inputRange_norm.keys()[5:]):
                # Threshold input rhow, in case of negative or too high value
                j_glob = iband_NN_norm[j]
                rhow_in = max(rhow[i, j_glob], np.exp(inputRange_norm[var][0]))
                rhow_in = min(rhow_in, np.exp(inputRange_norm[var][1]))
                if rhow_in != rhow[i, j_glob]:
                    FlagConstraintApplied[i] = 1
                inputNN[j+5] = np.log(rhow_in)
            log_rw_nn2 = np.array(nn_rw_rwnorm.calc(inputNN), dtype=np.float32)
            output[i, :] = np.exp(log_rw_nn2)

    return output, FlagConstraintApplied

def write_BalticP_AC_Product(product, baltic__product_path, sensor, spectral_dict, scalar_dict=None,
                             copyOriginalProduct=False, outputProductFormat="BEAM-DIMAP", addname='',
                             add_Idepix_Flags=False, idepixProduct=None, add_L2Flags=False, L2FlagArray=None,
                             add_Geometry=False):
    # Initialise the output product
    File = jpy.get_type('java.io.File')
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    bandShape = (height, width)

    dirname = os.path.dirname(baltic__product_path)
    outname, ext = os.path.splitext(os.path.basename(baltic__product_path))
    if outputProductFormat == "BEAM-DIMAP":
        baltic__product_path = os.path.join(dirname, outname + addname +'.dim')
    elif outputProductFormat == 'CSV':
        baltic__product_path = os.path.join(dirname, outname + addname +'.csv')

    balticPACProduct = Product('balticPAC', 'balticPAC', width, height)
    balticPACProduct.setFileLocation(File(baltic__product_path))

    ProductUtils.copyGeoCoding(product, balticPACProduct)

    # Define total number of bands (TOA)
    if (sensor == 'OLCI'):
        nbands = 21
    elif (sensor == 'S2MSI'):
        nbands = 13

    for key in spectral_dict.keys():
        data = spectral_dict[key].get('data')
        if not data is None:
            nbands_key = data.shape[-1]
            if outputProductFormat == 'BEAM-DIMAP':
                if sensor == 'OLCI':
                    bsources = [product.getBand("Oa%02d_radiance" % (i + 1)) for i in range(nbands)]
                elif sensor == 'S2MSI':
                    bsources = [product.getBand("B%d" % (i + 1)) for i in range(8)]
                    bsources.append(product.getBand('B8A'))
                    [bsources.append(product.getBand("B%d" % (i + 9))) for i in range(4)]

            for i in range(nbands_key):
                brtoa_name = key + "_" + str(i + 1)
                band = balticPACProduct.addBand(brtoa_name, ProductData.TYPE_FLOAT64)
                if outputProductFormat == 'BEAM-DIMAP':
                    ProductUtils.copySpectralBandProperties(bsources[i], band)
                band.setNoDataValue(np.nan)
                band.setNoDataValueUsed(True)

                sourceData = np.array(data[:, i], dtype='float64').reshape(bandShape)
                band.setRasterData(ProductData.createInstance(sourceData))


    # Create empty bands for scalar fields
    if not scalar_dict is None:
        for key in scalar_dict.keys():
            singleBand = balticPACProduct.addBand(key, ProductData.TYPE_FLOAT64)
            singleBand.setNoDataValue(np.nan)
            singleBand.setNoDataValueUsed(True)
            data = scalar_dict[key].get('data')
            if not data is None:
                sourceData = np.array(data, dtype='float64').reshape(bandShape)
                singleBand.setRasterData(ProductData.createInstance(sourceData))

    if copyOriginalProduct:
        originalBands = product.getBandNames()
        balticBands = balticPACProduct.getBandNames()
        for bb in balticBands:
            originalBands = [ob for ob in originalBands if ob != bb]
        for ob in originalBands:
            singleBand = balticPACProduct.addBand(ob, ProductData.TYPE_FLOAT64)
            singleBand.setNoDataValue(np.nan)
            singleBand.setNoDataValueUsed(True)

            data = get_band_or_tiePointGrid(product,ob)
            sourceData = np.array(data, dtype='float64').reshape(bandShape)
            singleBand.setRasterData(ProductData.createInstance(sourceData))

    if add_Idepix_Flags:
        flagBand = balticPACProduct.addBand('pixel_classif_flags', ProductData.TYPE_INT32)
        flagBand.setDescription('Idepix flag information')
        flagBand.setNoDataValue(np.nan)
        flagBand.setNoDataValueUsed(True)

        data = get_band_or_tiePointGrid(idepixProduct, 'pixel_classif_flags')
        sourceData = np.array(data, dtype='int32').reshape(bandShape)
        flagBand.setRasterData(ProductData.createInstance(sourceData))

        idepixFlagCoding = FlagCoding('pixel_classif_flags')
        flagNames = list(idepixProduct.getAllFlagNames())
        print(list(flagNames))
        IDflags = 'pixel_classif_flags'
        flagNames = [fn[(len(IDflags)+1):] for fn in flagNames if IDflags in fn]
        for i, fn in enumerate(flagNames):
            idepixFlagCoding.addFlag(fn, 2**i, fn)
        balticPACProduct.getFlagCodingGroup().add(idepixFlagCoding)
        flagBand.setSampleCoding(idepixFlagCoding)

    if add_L2Flags:
        flagBand = balticPACProduct.addBand('baltic_L2_flags', ProductData.TYPE_INT32)
        flagBand.setDescription('L2 flag information for the baltic+ AC')
        flagBand.setNoDataValue(np.nan)
        flagBand.setNoDataValueUsed(True)

        sourceData = np.array(L2FlagArray, dtype='int32').reshape(bandShape)
        flagBand.setRasterData(ProductData.createInstance(sourceData))

        L2FlagCoding = FlagCoding('baltic_L2_flags')
        flagNames = ['OOR_NN_IOP', 'OOR_NN_normalisation', 'NELDER_MEAD_FAIL']
        flagDescription = ['input IOPs to forwardNN out of range. at least one IOP has been constrained.',
                           'input rho_w to NormalisationNN out of range. at least one rho_w has been constrained.',
                           'Nelder-Mead Optimisation failed.']
        #IDflags = 'baltic_L2_flags'
        #flagNames = [fn[(len(IDflags) + 1):] for fn in flagNames if IDflags in fn]
        i = 0
        for fn, dscr in zip(flagNames, flagDescription):
            L2FlagCoding.addFlag(fn, 2 ** i, dscr)
            i += 1
        balticPACProduct.getFlagCodingGroup().add(L2FlagCoding)
        flagBand.setSampleCoding(L2FlagCoding)

    if add_Geometry:
        oaa, oza, saa, sza = angle_Reader(product, sensor)
        if sensor == 'OLCI':
            geomNames = ['OAA', 'OZA', 'SAA', 'SZA']
        elif sensor == 'S2MSI':
            geomNames = ['view_azimuth_mean', 'view_zenith_mean', 'sun_azimuth', 'sun_zenith']

        dataList = [oaa, oza, saa, sza]

        for gn, data in zip(geomNames, dataList):
            singleBand = balticPACProduct.addBand(gn, ProductData.TYPE_FLOAT64)
            singleBand.setNoDataValue(np.nan)
            singleBand.setNoDataValueUsed(True)

            sourceData = np.array(data, dtype='float64').reshape(bandShape)
            singleBand.setRasterData(ProductData.createInstance(sourceData))



    if outputProductFormat == 'BEAM-DIMAP':
        # Set auto grouping
        autoGroupingString = ':'.join(spectral_dict.keys())
        balticPACProduct.setAutoGrouping(autoGroupingString)

    GPF.writeProduct(balticPACProduct, File(baltic__product_path), outputProductFormat, False, ProgressMonitor.NULL)

    balticPACProduct.closeIO()


def polymer_matrix(bands_sat,bands,valid,rho_g,rho_r,sza,oza,wavelength,adf_ppp):
    """
    Compute matrices associated to the polynomial modelling of the atmosphere (POLYMER)
    Care: direct matrice (forward model) is for 'bands_sat', while inverse (backward model) is limited to 'bands'
    """

    # Define matrix of polynomial modelling (c0 T0 + c1 lambda^-1 + c2 rho_R)
    ncoef = 3
    nband = len(bands)
    nband_sat = len(bands_sat)
    npix = rho_g.shape[0]
    Aatm = np.zeros((npix, nband_sat, ncoef), dtype='float32') # Care, defined for bands_sat
    Aatm_inv = np.zeros((npix, ncoef, nband), dtype='float32') # Care, defined for only bands

    # Indices of bands_corr in all bands
    iband = np.searchsorted(bands_sat, bands)

    # Compute T0
    taum = 0.00877*((wavelength/1000.)**(-4.05))
    air_mass = 1./np.cos(np.radians(sza))+1./np.cos(np.radians(oza))
    rho_g0 = 0.02
    factor = (1-0.5*np.exp(-rho_g/rho_g0))*air_mass
    T0 = np.exp(-taum*factor[:,None])

    Aatm[:,:,0] = T0
    Aatm[:,:,1] = (wavelength/1000.)**-1.
    Aatm[:,:,2] = rho_r

    # Compute pseudo inverse: A* = ((A'.A)^(-1)).A' /!\ limited to bands
    Aatm_corr = Aatm[:,iband,:]
    Aatm_inv[valid] = np.linalg.pinv(Aatm_corr[valid])

    return Aatm, Aatm_inv

def check_and_constrain_iop(log_iop, inputRange,sensor):
    for i in range(len(log_iop)):
        var = inputRange.keys()[5+i] # discard first variables (angles, temperature, salinity)
        mi = inputRange[var][0]
        ma = inputRange[var][1]
        if log_iop[i] < mi:
            log_iop[i] = mi
        if log_iop[i] > ma:
            log_iop[i] = ma
    return log_iop


def ac_cost(log_iop, sensor, nbands, iband_NN, iband_corr, iband_chi2, rho_rc, td, sza, oza, raa, Aatm, Aatm_inv, valid, inputRange_forward):
    """
    Cost function to be minimized, define for one pixel
    """

    # Compute rhow_mod
    rho_wmod = np.zeros(nbands) + np.NaN

    # Check iop range and apply constraints to forwardNN input range
    log_iop = check_and_constrain_iop(log_iop, inputRange_forward, sensor)

    rho_wmod[iband_NN] = apply_forwardNN_IOP_to_rhow(np.array([log_iop]), np.array([sza]), np.array([oza]), np.array([raa]), sensor,np.array([valid]),iband_NN)
    # Compute rho_ag and fit best atmospheric model
    rho_ag = rho_rc - td*rho_wmod
    coefs = np.einsum('...ij,...j->...i', Aatm_inv, rho_ag[iband_corr])
    rho_ag_mod = np.einsum('...ij,...j->...i',Aatm,coefs)
    # Compute rho_w
    trho_w = (rho_rc - rho_ag_mod)#/td
    # Compute residual and chi2
    res  = trho_w[iband_chi2]-td[iband_chi2]*rho_wmod[iband_chi2] # TODO one other option is to minimize at TOA by multiplying by td
    chi2 = np.sum(res*res) # TODO cost function should include weighting; option in relative difference
    return chi2


def baltic_AC_forwardNN(scene_path='', filename='', outpath='', sensor='', subset=None, addName = '', outputSpectral=None,
                        outputScalar=None, correction='HYGEOS', copyOriginalProduct=False, outputProductFormat="BEAM-DIMAP",
                        atmosphericAuxDataPath = None, niop=5, add_Idepix_Flags=True, add_L2Flags=False, add_c2rccIOPs=False, NNversion='v3'):
    """
    Main function to run the Baltic+ AC based on forward NN
    correction: 'HYGEOS' or 'IPF' for Rayleigh+glint correction
    """

    # Read the NNs
    read_NNs(sensor, NNversion)

    # Read NNs input range
    inputRange_forward = read_NN_input_ranges_fromFile(nnForwardFilePath)
    inputRange_backward = read_NN_input_ranges_fromFile(nnBackwardFilePath)
    inputRange_norm = read_NN_input_ranges_fromFile(nnNormFilePath)

    # Get sensor & AC bands
    bands_sat, bands_rw, bands_corr, bands_chi2, bands_forwardNN, bands_backwardNN, bands_normNN, bands_abs = get_bands.main(sensor,"dummy")
    nbands = len(bands_sat)
    iband_corr = np.searchsorted(bands_sat, bands_corr)
    iband_chi2 = np.searchsorted(bands_sat, bands_chi2)
    iband_NN_forward = np.searchsorted(bands_sat, bands_forwardNN)
    iband_NN_backward = np.searchsorted(bands_sat, bands_backwardNN)
    iband_NN_norm = np.searchsorted(bands_sat, bands_normNN)
    iband_abs = np.searchsorted(bands_sat, bands_abs)

    # Initialising a product for Reading with snappy
    product = snp.ProductIO.readProduct(os.path.join(scene_path, filename))

    # Resampling S2MSI to 60m
    if sensor == "S2MSI" and product.isMultiSize():
        print( "Resample MSI data")
        parameters = HashMap()
        parameters.put('resolution', '60')
        parameters.put('upsampling', 'Bicubic')
        parameters.put('downsampling', 'Mean') #
        parameters.put('flagDownsampling', 'FlagOr') # 'First', 'FlagAnd'
        product = GPF.createProduct('S2Resampling', parameters, product)


    # Get scene size
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    npix = width*height

    # Read L1B product and convert Radiance to reflectance
    rho_toa = radianceToReflectance_Reader(product, sensor=sensor)

    # Classify pixels with Level-1 flags
    valid = check_valid_pixel_expression_L1(product, sensor)
    print(np.sum(valid), len(valid))

    if add_Idepix_Flags:
        idepixProduct = run_IdePix_processor(product, sensor)
        validIdepix = check_valid_pixel_expression_Idepix(idepixProduct, sensor)
        valid = np.logical_and(valid, validIdepix)
    else:
        idepixProduct=None


    # Limit processing to sub-box
    if subset: #FIXME should be only applied to input raster file
         sline,eline,scol,ecol = subset
         valid_subset = np.zeros((height,width),dtype='bool')
         valid_subset[sline:eline+1,scol:ecol+1] = valid.reshape(height,width)[sline:eline+1,scol:ecol+1]
         valid = valid_subset.reshape(height*width)
         del valid_subset
    
    # Read geometry and compute relative azimuth angle
    oaa, oza, saa, sza = angle_Reader(product, sensor)
    raa, nn_raa = calculate_diff_azim(oaa, saa)

    # Read wavelength (per-pixel for OLCI) and geolocation
    if sensor == 'OLCI':
        # Read per-pixel wavelength
        wavelength = Level1_Reader(product, sensor, 'lambda0', reshape=False)
        # Read latitude, longitude
        latitude = get_band_or_tiePointGrid(product, 'latitude', reshape=False)
        longitude = get_band_or_tiePointGrid(product, 'longitude', reshape=False)
    elif sensor == 'S2MSI':
        # Duplicate wavelengths for all pixels
        wavelength = np.tile(bands_sat,(len(sza),1)) #TODO: integrate band with S2 SRF
        # Read latitude, longitude
        latitude, longitude = getGeoPositionsForS2Product(product)

    # Read day in the year
    yday = get_yday(product, reshape=False)

    # Read meteo data
    print( "Read meteo data")
    if sensor == 'OLCI':
        pressure = get_band_or_tiePointGrid(product, 'sea_level_pressure', reshape=False)
        ozone = get_band_or_tiePointGrid(product, 'total_ozone', reshape=False)
        tcwv = get_band_or_tiePointGrid(product, 'total_columnar_water_vapour', reshape=False)
        wind_u = get_band_or_tiePointGrid(product, 'horizontal_wind_vector_1', reshape=False)
        wind_v = get_band_or_tiePointGrid(product, 'horizontal_wind_vector_2', reshape=False)
        windm = np.sqrt(wind_u*wind_u+wind_v*wind_v)
        altitude = get_band_or_tiePointGrid(product, 'altitude', reshape=False)
    elif sensor == 'S2MSI':
        bandNames = list(product.getBandNames())
        ## only for match-up data with included meteorology.
        if ('pressure' in bandNames) and ('ozone' in bandNames) and ('tcwv' in bandNames)\
                and ('wind_u' in bandNames) and ('wind_v' in bandNames):
            pressure = get_band_or_tiePointGrid(product, 'pressure', reshape=False)
            ozone = get_band_or_tiePointGrid(product, 'ozone', reshape=False)
            tcwv = get_band_or_tiePointGrid(product, 'tcwv', reshape=False)
            wind_u = get_band_or_tiePointGrid(product, 'wind_u', reshape=False)
            wind_v = get_band_or_tiePointGrid(product, 'wind_v', reshape=False)
            windm = np.sqrt(wind_v * wind_v + wind_u * wind_u)
        else:
            if atmosphericAuxDataPath != None:
                # Compute aux data (one unique value per scene)
                AuxFullFilePath_dict = checkAuxDataAvailablity(atmosphericAuxDataPath, product=product)
                AuxDataDict = setAuxData(product, AuxFullFilePath_dict)
                # Apply values to the whole image
                shape = sza.shape
                pressure = np.ones(shape)*AuxDataDict['pressure']
                ozone = np.ones(shape)*AuxDataDict['ozone']
                tcwv = np.ones(shape)*AuxDataDict['tcwv']
                wind_u = np.ones(shape)*AuxDataDict['wind_u']
                wind_v = np.ones(shape)*AuxDataDict['wind_v']
                windm = np.sqrt(wind_v*wind_v + wind_u*wind_u)
            else:
                print('Please set a path to the AUX data archive.')
                sys.exit(1)

    # Read LUTs
    if sensor == 'OLCI':
        file_adf_acp = default_ADF['OLCI']['file_adf_acp']
        file_adf_ppp = default_ADF['OLCI']['file_adf_ppp']
        file_adf_clp = default_ADF['OLCI']['file_adf_clp']
        adf_acp = luts_olci.LUT_ACP(file_adf_acp)
        adf_ppp = luts_olci.LUT_PPP(file_adf_ppp)
        adf_clp = luts_olci.LUT_CLP(file_adf_clp)
        if correction == 'HYGEOS':
            LUT_HYGEOS = lut_hygeos.LUT(default_ADF['OLCI']['file_HYGEOS'])
    #elif sensor == 'S2' TODO

    print("Pre-corrections")
    # Gaseous correction
    rho_ng = gas_correction(rho_toa, valid, latitude, longitude, yday, sza, oza, raa, wavelength,
            pressure, ozone, tcwv, adf_ppp, adf_clp, sensor)

    # Vicarious calibration
    #rho_ng = vicarious_calibration(rho_ng, valid, adf_acp, sensor)

    # Compute diffuse transmittance (Rayleigh)
    td = diffuse_transmittance(sza, oza, pressure, adf_ppp)

    # Glint correction - rho_g required even for HYGEOS correction
    rho_g, rho_gc = glint_correction(rho_ng, valid, sza, oza, saa, raa, pressure, wind_u, wind_v, windm, adf_ppp)

    if correction == 'IPF':
        # White-caps correction
        #rho_wc, rho_gc = white_caps_correction(rho_ng, valid, windm, td, adf_ppp)

        # Rayleigh correction
        rho_r, rho_rc = Rayleigh_correction(rho_gc, valid, sza, oza, raa, pressure, windm, adf_acp, adf_ppp, sensor)

    elif correction == 'HYGEOS':
        # Glint + Rayleigh correction
        rho_r, rho_molgli, rho_rc, tau_r, tau_r_mono = Rmolgli_correction_Hygeos(rho_ng, valid, latitude, sza, oza, raa, wavelength,
                                                                     pressure, windm, LUT_HYGEOS, altitude)

    # Atmospheric model
    print("Compute atmospheric matrices")
    Aatm, Aatm_inv = polymer_matrix(bands_sat,bands_corr,valid,rho_g,rho_r,sza,oza,wavelength,adf_ppp)
    l2flags = np.zeros(npix, dtype='int32')

    # Inversion of log_iop = [log_apig, log_adet, log a_gelb, log_bpart, log_bwit]
    print("Inversion")
    log_iop = np.zeros((npix,niop)) + np.NaN
    percent_old = 0
    ipix_proc = 0
    npix_proc = np.count_nonzero(valid)


    for ipix in range(npix):
        if not valid[ipix]: continue
        # Display processing progress with respect to the valid pixels
        percent = (int(float(ipix_proc)/float(npix_proc)*100)/10)*10
        if percent != percent_old:
            percent_old = percent
            sys.stdout.write(" ...%d%%"%percent)
            sys.stdout.flush()
        # First guess
        #log_iop_0 = np.array([-4.3414865, -4.956355, - 3.7658699, - 1.8608053, - 2.694404])
        log_iop_0 = np.zeros(niop) -3. # array([-3., -3., -3., -3., -3.])
        # Nelder-Mead optimization
        args_pix = (sensor, nbands, iband_NN_forward, iband_corr, iband_chi2, rho_rc[ipix], td[ipix], sza[ipix], oza[ipix], nn_raa[ipix], Aatm[ipix], Aatm_inv[ipix], valid[ipix], inputRange_forward)
        NM_res = minimize(ac_cost, log_iop_0, args=args_pix, method='nelder-mead')#, options={'maxiter':150', disp': True})
        log_iop[ipix,:] = NM_res.x
        success = NM_res.success
        if not success:
            l2flags[ipix] += 2 ** 2
        ipix_proc += 1
    print("")

    # Compute the final residual
    print("Compute final residual")
    rho_wmod = np.zeros((npix, nbands)) + np.NaN
    rho_wmod[:, iband_NN_forward] = apply_forwardNN_IOP_to_rhow(log_iop, sza, oza, nn_raa, sensor, valid, iband_NN_forward)
    rho_ag_mod = np.zeros((npix, nbands)) + np.NaN
    rho_ag = rho_rc - td*rho_wmod
    coefs = np.einsum('...ij,...j->...i', Aatm_inv[valid], rho_ag[valid][:,iband_corr])
    rho_ag_mod[valid] = np.einsum('...ij,...j->...i',Aatm[valid],coefs)
    rho_w = (rho_rc - rho_ag_mod)/td

    # Set absorption band to NaN
    rho_ag[:,iband_abs] = np.NaN
    rho_ag_mod[:,iband_abs] = np.NaN
    rho_rc[:,iband_abs] = np.NaN
    rho_w[:,iband_abs] = np.NaN

    # Apply normalisation
    print("Normalize spectra")
    rho_wn, oorFlagArray = apply_NN_rhow_to_rhownorm(rho_w, sza, oza, nn_raa, sensor, valid, nn_rw_rwnorm, iband_NN_norm, inputRange_norm)

    l2flags[np.array(oorFlagArray==1)] += 2**1

    # Compute IOPs if requested
    if add_c2rccIOPs:
        log_iop_c2rcc, oorBackwardNN = apply_backwardNN_rhow_to_IOP(rho_w, sza, oza, nn_raa, sensor, valid, nn_rw_iop, iband_NN_backward, inputRange_backward)

    # TODO uncertainties
    # unc_rhow =

    ###
    # Writing a product
    # input: spectral_dict holds the spectral fields
    #        scalar_dict holds the scalar fields
    ###
    baltic__product_path = os.path.join(outpath,'baltic_' + filename)
    if outputSpectral:
        spectral_dict = {}
        for field in outputSpectral.keys():
            spectral_dict[field] = {'data': eval(outputSpectral[field])}
    else: spectral_dict = None

    if outputScalar:
        scalar_dict = {}
        for field in outputScalar.keys():
            scalar_dict[field] = {'data': eval(outputScalar[field])}
    else: scalar_dict = None

    if add_c2rccIOPs:
        iop_names = ['apig', 'adet', 'a_gelb', 'bpart', 'bwit']
        if scalar_dict is None:
            scalar_dict = {}
        for i, field in enumerate(iop_names):
            scalar_dict[field] = {'data': np.exp(log_iop_c2rcc[:,i])}

    write_BalticP_AC_Product(product, baltic__product_path, sensor, spectral_dict, scalar_dict,
                             copyOriginalProduct, outputProductFormat, addName,
                             add_Idepix_Flags=add_Idepix_Flags, idepixProduct=idepixProduct,
                             add_L2Flags=add_L2Flags, L2FlagArray=l2flags,
                             add_Geometry=True)

    product.closeProductReader()



