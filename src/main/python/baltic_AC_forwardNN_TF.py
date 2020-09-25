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
from scipy.optimize import curve_fit, least_squares, minimize
import sys
import time
import glob
import tensorflow as tf

# snappy import
import snappy as snp
from snappy import Product
from snappy import ProductData
#from snappy import ProductDataUTC
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

# POLYMER
sys.path.append('/home/cmazeran/Documents/solvo/Projets/Brockmann_Consult/OC_CCI/POLYMER4.11')
from polymer.water import ParkRuddick
path_aux_common = '/home/cmazeran/Documents/solvo/Projets/Brockmann_Consult/OC_CCI/POLYMER4.1/auxdata/common/'

# Set locale for proper time reading with datetime
locale.setlocale(locale.LC_ALL, 'en_US.UTF_8')


# Define NNs as global variables
def read_NNs(sensor, NNversion):
    global nnForwardFilePath, nnBackwardFilePath, nnNormFilePath
    global nn_forward, nn_backward
    global inputRange_forward
    global nn_iop_rw, nn_rw_iop
    global session

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
        elif NNversion == 'TF':
            nnForwardFilePath = "NN_TF/fwNNHL_logLossI8x50x40x40xO16_auxVarN0_batch20000_epoch60000_loss0.0.h5"
            nnBackwardFilePath = "NN_TF/bwNNHL_LossLogRhow_I8x50x40x40xO16_b2E5_e6E5_I19x77x77x77xO5_auxVarN0_batch10000_epoch90000_loss0.0.h5"
        elif NNversion == 'TF_Schiller': #input + output like 'TF'; converted into old .net format and readable with old c-functions, which might be faster than tensorflow??
            nnForwardFilePath = "NN_reciprocal/fwNNc2rcc_linearLossI8x97x97x97xO12batch300_epoch200000_loss0.0.net"
            nnBackwardFilePath = "NN_reciprocal/bwNNc2rcc_LossLogRhow_fwNN97_2e6_I15x77x77x77xO5batch300_epoch100000_loss0.00242.net"

    elif sensor == 'S2MSI':
        nnForwardFilePath = "forwardNN_c2rcc/msi/std_s2_20160502/iop_rw/17x97x47_125.5.net"
        nnBackwardFilePath = "forwardNN_c2rcc/msi/std_s2_20160502/rw_rwnorm/27x7x27_28.0.net"
        nnNormFilePath = "forwardNN_c2rcc/msi/std_s2_20160502/rw_rwnorm/27x7x27_28.0.net"

    if NNversion == 'v2' or NNversion == 'v3':
        inputRange_forward = read_NN_input_ranges_fromFile(nnForwardFilePath)
        inputRange_backward = read_NN_input_ranges_fromFile(nnBackwardFilePath)
        # TODO READ NNs
    elif NNversion == 'TF':
        # Open session for tf v1.X
        TF_version = tf.__version__.split('.')[0]
        if TF_version == '1':
            session = tf.InteractiveSession()
        else:
            session = None
        # Read NNs
        nn_forward = tf.keras.models.load_model(nnForwardFilePath)
        nn_backward = tf.keras.models.load_model(nnBackwardFilePath)
        #col_names = ['apig', 'adet', 'agelb', 'bpart', 'bwit']
        inputRange_forward = { #TODO OrderedDict?
            'apig' :(-4.482068, 0.82549036),
            'adet':(-4.828189, 1.3987169),
            'agelb': (-2.5902672, 1.9878744),
            'bpart': (-2.690456, 4.9607444),
            'bwit':  (-3.2042336, 5.568726),
            'nn_raa':   (-1.0, 1.0),
            'oza': (0.49999997, 1.0),
            'sza': (0.25881895, 1.0)
        }
        # TODO inputRange_backward
    elif NNversion == 'TF_Schiller':
        inputRange_forward = { #TODO OrderedDict?
            'apig': (-4.431202411651611, 0.8628255128860474),
            'adet': (-4.135058403015137, 2.091784954071045),
            'agelb': (-1.897108793258667, 2.6810054779052734),
            'bpart': (-2.7132925987243652, 4.960668087005615),
            'bwit': (-3.204205274581909, 5.568716526031494),
            'nn_raa': (-1.0, 1.0),
            'oza': (0.573578953742981, 1.0),
            'sza': (0.25882574915885925, 1.0)
        }
        # TODO inputRange_backward
        NNffbpAlphaTabFast = jpy.get_type('org.esa.snap.core.nn.NNffbpAlphaTabFast')
        nnfile = open(nnForwardFilePath, 'r')
        nnCode = nnfile.read()
        nn_iop_rw = NNffbpAlphaTabFast(nnCode)
        nnfile.close()
        nnfile = open(nnBackwardFilePath, 'r')
        nnCode = nnfile.read()
        nn_rw_iop = NNffbpAlphaTabFast(nnCode)
        nnfile.close()

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

def get_band_or_tiePointGrid(product, name, dtype='float32', reshape=True, subset=None):
    ##
    # This function reads a band or tie-points, identified by its name <name>, from SNAP product <product>
    # The fuction returns a numpy array of shape (height, width)
    ##
    if subset is None:
        height = product.getSceneRasterHeight()
        width = product.getSceneRasterWidth()
        sline, eline, scol, ecol = 0, height-1, 0, width -1
    else:
        sline,eline,scol,ecol = subset
        height = eline - sline + 1
        width = ecol - scol + 1

    var = np.zeros(width * height, dtype=dtype)
    if name in list(product.getBandNames()):
        product.getBand(name).readPixels(scol, sline, width, height, var)
    elif name in list(product.getTiePointGridNames()):
        var.shape = (height, width)
        for i,iglob in enumerate(range(sline,eline+1)):
            for j,jglob in enumerate(range(scol,ecol+1)):
                var[i, j] = product.getTiePointGrid(name).getPixelDouble(jglob, iglob)
        var.shape = (height*width)
    else:
        raise Exception('{}: neither a band nor a tie point grid'.format(name))

    if reshape:
        var.shape = (height, width)

    return var

def Level1_Reader(product, sensor, band_group='radiance', reshape=True, subset=None):
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
    if subset is None:
        height = product.getSceneRasterHeight()
        width = product.getSceneRasterWidth()
    else:
        sline,eline,scol,ecol = subset
        height = eline - sline + 1
        width = ecol - scol + 1
    var = np.zeros((width * height, len(input_label)))
    for i, bn in enumerate(input_label):
        var[:,i] = get_band_or_tiePointGrid(product, bn, reshape=reshape, subset=subset)

    return var

def get_yday(product,reshape=True, subset=None):
    # Get product size
    height_full = product.getSceneRasterHeight()
    width_full = product.getSceneRasterWidth()
    if subset is None:
        height = height_full
        width = width_full
    else:
        sline,eline,scol,ecol = subset
        height = eline - sline + 1
        width = ecol - scol + 1

    ## time handling for match-up files
    if str(product.getFileLocation()).endswith('.txt') or str(product.getFileLocation()).endswith('.csv'):
        if not subset is None:
            print('Error: subset option not compatible with match-up file')
            sys.exit(1)
        yday = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                a = product.getSceneTimeCoding().getMJD(PixelPos(x + 0.5, y + 0.5))
                year, yday[y, x], hour = yearAndDoyAndHourUTC(a)

    else:
        ## time handling for a scene
        # Get yday of each row (full scene)
        dstart = datetime.datetime.strptime(str(product.getStartTime()),'%d-%b-%Y %H:%M:%S.%f')
        dstop = datetime.datetime.strptime(str(product.getEndTime()),'%d-%b-%Y %H:%M:%S.%f')
        dstart = np.datetime64(dstart)
        dstop = np.datetime64(dstop)
        dpix = dstart + (dstop-dstart)/float(height_full-1.)*np.arange(height_full)
        yday = [k.timetuple().tm_yday for k in dpix.astype(datetime.datetime)]
        # Limit to subset
        if subset:
            yday = yday[sline:eline+1]

        # Apply to all columns
        yday = np.array(yday*width).reshape(width,height).transpose()

    if not reshape:
        yday = np.ravel(yday)

    return yday

def radianceToReflectance_Reader(product, sensor = '', print_info=False, subset=None):

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
        rad = Level1_Reader(product, sensor, band_group='radiance',reshape=False, subset=subset)
        solar_flux = Level1_Reader(product, sensor, band_group='solar_flux',reshape=False, subset=subset)
        SZA = get_band_or_tiePointGrid(product, 'SZA', reshape=False, subset=subset)
        refl = rad * np.pi / (solar_flux * np.cos(SZA.reshape(rad.shape[0],1)*np.pi/180.))
    elif sensor == 'S2MSI':
        refl = Level1_Reader(product, sensor,reshape=False, subset=subset)

    return refl

def angle_Reader(product, sensor, subset=None):
    if sensor == 'OLCI':
        oaa = get_band_or_tiePointGrid(product, 'OAA', reshape=False, subset=subset)
        oza = get_band_or_tiePointGrid(product, 'OZA', reshape=False, subset=subset)
        saa = get_band_or_tiePointGrid(product, 'SAA', reshape=False, subset=subset)
        sza = get_band_or_tiePointGrid(product, 'SZA', reshape=False, subset=subset)
    elif sensor == 'S2MSI':
        oaa = get_band_or_tiePointGrid(product, 'view_azimuth_mean', reshape=False, subset=subset)
        oza = get_band_or_tiePointGrid(product, 'view_zenith_mean', reshape=False, subset=subset)
        saa = get_band_or_tiePointGrid(product, 'sun_azimuth', reshape=False, subset=subset)
        sza = get_band_or_tiePointGrid(product, 'sun_zenith', reshape=False, subset=subset)

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

def check_valid_pixel_expression_Idepix(product, sensor, subset=None):
    valid_pixel_flag = None

    if subset is None:
        height = product.getSceneRasterHeight()
        width = product.getSceneRasterWidth()
        sline = 0
        scol = 0
    else:
        sline,eline,scol,ecol = subset
        height = eline - sline + 1
        width = ecol - scol + 1

    quality_flags = np.zeros(width * height, dtype='uint32')
    product.getBand('pixel_classif_flags').readPixels(scol, sline, width, height, quality_flags)

    if sensor == 'OLCI':
        # Masks OLCI Idepix
        invalid_mask = np.bitwise_and(quality_flags, 2 ** 0) == 2 ** 0
        cloud_mask = np.bitwise_and(quality_flags, 2 ** 1) == 2 ** 1
        cloudbuffer_mask = np.bitwise_and(quality_flags, 2 ** 4) == 2 ** 4
        snowice_mask = np.bitwise_and(quality_flags, 2 ** 6) == 2 ** 6

        invalid_mask = np.logical_or(np.logical_or(invalid_mask, snowice_mask), np.logical_or(cloud_mask, cloudbuffer_mask))
        valid_pixel_flag = np.logical_not(invalid_mask)

    if sensor == 'S2MSI':
        # Masks S2MSI Idepix
        invalid_mask = np.bitwise_and(quality_flags, 2 ** 0) == 2 ** 0
        cloud_mask = np.bitwise_and(quality_flags, 2 ** 1) == 2 ** 1
        cloudbuffer_mask = np.bitwise_and(quality_flags, 2 ** 4) == 2 ** 4
        snowice_mask = np.bitwise_and(quality_flags, 2 ** 6) == 2 ** 6

        invalid_mask = np.logical_or(np.logical_or(invalid_mask, snowice_mask),
                                     np.logical_or(cloud_mask, cloudbuffer_mask))
        valid_pixel_flag = np.logical_not(invalid_mask)

    return valid_pixel_flag


def check_valid_pixel_expression_L1(product, sensor, subset=None):
    valid_pixel_flag = None

    if subset is None:
        height = product.getSceneRasterHeight()
        width = product.getSceneRasterWidth()
        sline = 0
        scol = 0
    else:
        sline,eline,scol,ecol = subset
        height = eline - sline + 1
        width = ecol - scol + 1

    if sensor == 'OLCI':
        quality_flags = np.zeros(width * height, dtype='uint32')

        # match-up extractions from Calvalus do not have a 'quality_flags' band
        if product.getBand('quality_flags') is None:
            invalid_mask = np.zeros(width * height, dtype='uint32')
            product.getBand('quality_flags.invalid').readPixels(scol, sline, width, height, invalid_mask)
            land_mask = np.zeros(width * height, dtype='uint32')
            product.getBand('quality_flags.land').readPixels(scol, sline, width, height, land_mask)
            coastline_mask =  np.zeros(width * height, dtype='uint32')
            product.getBand('quality_flags.coastline').readPixels(scol, sline, width, height, coastline_mask)
            inland_mask =  np.zeros(width * height, dtype='uint32')
            product.getBand('quality_flags.fresh_inland_water').readPixels(scol, sline, width, height, inland_mask)

            land_mask = coastline_mask | (land_mask & ~inland_mask)

            bright_mask = np.zeros(width * height, dtype='uint32')
            product.getBand('quality_flags.bright').readPixels(scol, sline, width, height, bright_mask)

            invalid_mask = np.logical_or(invalid_mask, np.logical_or(land_mask, bright_mask))
            valid_pixel_flag = np.logical_not(invalid_mask)
        else:
            product.getBand('quality_flags').readPixels(scol, sline, width, height, quality_flags)

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
        valid_pixel_flag = np.ones(width * height, dtype='uint32')

        b8 = get_band_or_tiePointGrid(product, 'B8', reshape=False, subset=subset)
        valid_pixel_flag = np.logical_and(np.array(b8 > 0), np.array(b8 < 0.1))

    return valid_pixel_flag


def apply_forwardNN_newNN_oldFormat(log_iop, sun_zenith, view_zenith, diff_azimuth, sensor, valid_data, iband_forwardNN):
    """
        Apply the forwardNN: IOP to rhow (trained on c2rcc simulations)
        input: numpy array log_iop, shape = (Npixels x log_iops= (log_apig, log_adet, log a_gelb, log_bpart, log_bwit)),
                np.array raa, shape = (Npixels,) range: 0-180
                np.array oza, shape = (Npixels,)
                np.array sza, shape = (Npixels,);
        returns: np.array rhow, shape = (Npixels, wavelengths)
        valid ranges can be found at the beginning of the .net-file.
        NN output for OLCI (12 bands): log_rw at lambda = 400, 412, 443, 489, 510, 560, 620, 665, 674, 681, 709, 754
        NN output for S2MSI (6 bands): log_rw at lambda = 443, 490, 560, 665, 705, 740
    """

    # Initialise output
    nBands = len(iband_forwardNN)
    rhow = np.zeros((log_iop.shape[0], nBands)) + np.NaN

    # Prepare the NN
    NN_input = np.zeros((log_iop.shape[0], 5 + 3)) - 1.  # IOPS + angles CARE -1 used by default for IOP when niop < 5
    for i in range(log_iop.shape[1]):
        NN_input[:, i] = log_iop[:, i]
    NN_input[:, 7] = np.cos(sun_zenith * np.pi / 180.)
    NN_input[:, 6] = np.cos(view_zenith * np.pi / 180.)
    NN_input[:, 5] = np.cos(diff_azimuth * np.pi / 180.)

    for i in range(log_iop.shape[0]):
        if valid_data[i]:
            log_rw_nn2 = np.array(nn_iop_rw.calc(NN_input[i,:]), dtype=np.float32)
            rhow[i, :] = np.exp(log_rw_nn2)

    return rhow

def apply_backwardNN_newNN_oldFormat(rhow, sun_zenith, view_zenith, diff_azimuth, sensor, valid_data):
    """
    Apply the backwardNN: rhow to IOP (trained on c2rcc simulations)
    input: numpy array rhow, shape = (Npixels x rhow = (log_rw_band1, log_rw_band2_.... )),
            np.array raa, shape = (Npixels,); range: 0-180
            np.array oza, shape = (Npixels,)
            np.array sza, shape = (Npixels,)
    returns: np.array log_iop, shape = (Npixels x log_iop= (log_apig, log_adet, log a_gelb, log_bpart, log_bwit))
    valid ranges can be found at the beginning of the .net-file.
    NN input for OLCI (12 bands): log_rw at lambda = 400, 412, 443, 489, 510, 560, 620, 665, 674, 681, 709, 754
    NN input for S2MSI (8 bands): log_rw at lambda = 443, 490, 560, 665, 705, 740, 783, 865
    """

    # Initialise output
    log_iop = np.zeros((rhow.shape[0], 5)) + np.NaN

    # Prepare the NN, limit to rhow >0
    rhow_pos = np.all(rhow > 0, axis=1)
    valid2 = valid_data & rhow_pos
    NN_input = np.zeros((rhow.shape[0], rhow.shape[1] + 3), dtype='float32') + np.NaN  # cos angles + log(rhow)
    NN_input[:, 0] = np.cos(sun_zenith * np.pi / 180.)
    NN_input[:, 1] = np.cos(view_zenith * np.pi / 180.)
    NN_input[:, 2] = np.cos(diff_azimuth * np.pi / 180.)
    NN_input[:, 3:] = np.log(rhow)

    for i in range(rhow.shape[0]):
        if valid2[i]:
            # for j,var in enumerate(list(inputRange_backward.keys())[3:]):
            #     # Threshold input rhow, in case of negative or too high value
            #     j_glob = iband_backwardNN[j]
            #     # inputRange=list(inputRange_backward.keys())[5:][j]
            #     rhow_in = max(rhow[i, j_glob], np.exp(inputRange_backward[var][0]))
            #     rhow_in = min(rhow_in, np.exp(inputRange_backward[var][1]))
            #     if rhow_in != rhow[i, j_glob]:
            #         FlagConstraintApplied[i] = 1
            #     inputNN[j+5] = np.log(rhow_in)
            log_iop[i,:] = np.array(nn_rw_iop.calc(NN_input[i, :]), dtype=np.float32) # output: log_apig, log_adet, log a_gelb, log_bpart, log_bwit

    return log_iop


def apply_forwardNN(log_iop, sun_zenith, view_zenith, diff_azimuth, valid):
    """
    Apply the forwardNN: IOP to rhow
    input: numpy array log_iop, shape = (Npixels x log_iops= (log_apig, log_adet, log a_gelb, log_bpart, log_bwit)),
            np.array sza, shape = (Npixels,)
            np.array oza, shape = (Npixels,)
            np.array raa, shape = (Npixels,); range: 0-180
    returns: np.array rhow, shape = (Npixels, wavelengths)
    valid ranges can be found at the beginning of the .net-file.
    NN output for OLCI (12 bands): log_rw at lambda = 400, 412, 443, 489, 510, 560, 620, 665, 674, 681, 709, 754
    NN output for S2MSI (6 bands): log_rw at lambda = 443, 490, 560, 665, 705, 740
    """

    # Initialise output
    nBands = len(iband_forwardNN)
    rhow = np.zeros((log_iop.shape[0], nBands)) + np.NaN

    # Prepare the NN
    NN_input = np.zeros((log_iop.shape[0],5+3)) -1. # IOPS + angles CARE -1 used by default for IOP when niop < 5
    NN_input[:,0] = np.cos(sun_zenith * np.pi / 180.)
    NN_input[:,1] = np.cos(view_zenith * np.pi / 180.)
    NN_input[:,2] = np.cos(diff_azimuth * np.pi / 180.)
    for i in range(log_iop.shape[1]):
        NN_input[:,3+i] = log_iop[:,i]

    # Limit to valid pixels
    NN_input = np.array(NN_input[valid, :], dtype='float32')

    # Launch NN
    if session is not None:
        log_rhow = nn_forward(NN_input).eval(session=session)
    else:
        log_rhow = nn_forward(NN_input)

    # Get NN output
    for i in range(nBands):
        rhow[valid,i] = np.exp(log_rhow[:,i])

    return rhow

def apply_backwardNN(rhow, sun_zenith, view_zenith, diff_azimuth, valid):
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
    log_iop = np.zeros((rhow.shape[0], 5)) + np.NaN

    # Prepare the NN, limit to rhow >0
    rhow_pos = np.all(rhow>0, axis=1)
    valid2 = valid & rhow_pos
    nvalid = np.count_nonzero(valid2)
    NN_input = np.zeros((nvalid, rhow.shape[1] + 3), dtype='float32') # angles + log(rhow)
    NN_input[:,0] = np.cos(sun_zenith[valid2] * np.pi / 180.)
    NN_input[:,1] = np.cos(view_zenith[valid2] * np.pi / 180.)
    NN_input[:,2] = np.cos(diff_azimuth[valid2] * np.pi / 180.)
    NN_input[:,3:] = np.log(rhow[valid2,:])

    # Launch the NN
    if session is not None:
        log_iop[valid2,:] = nn_backward(NN_input).eval(session=session)
    else:
        log_iop[valid2,:] = nn_backward(NN_input)

    return log_iop



def final_setup_BalticProduct(product, baltic__product_path, sensor, spectral_dict, scalar_dict=None,
                              copyOriginalProduct=False, addname='',
                              add_Idepix_Flags=False, idepixProduct=None, add_L2Flags=False, L2FlagArray=None,
                              add_Geometry=False, subset=None):
    # Initialise the output product
    File = jpy.get_type('java.io.File')
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    bandShape = (height, width)

    if subset is None:
        height_subset = height
        width_subset = width
        sline, eline, scol, ecol = 0, height-1, 0, width -1
    else:
        sline,eline,scol,ecol = subset
        height_subset = eline - sline + 1
        width_subset = ecol - scol + 1
    bandShape_subset = (height_subset, width_subset)


    balticPACProduct = Product('balticPAC', 'balticPAC', width, height)
    #balticPACProduct.setFileLocation(File("E:\work\\test_from323.dim"))

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
            if sensor == 'OLCI':
                bsources = [product.getBand("Oa%02d_radiance" % (i + 1)) for i in range(nbands)]
            elif sensor == 'S2MSI':
                bsources = [product.getBand("B%d" % (i + 1)) for i in range(8)]
                bsources.append(product.getBand('B8A'))
                [bsources.append(product.getBand("B%d" % (i + 9))) for i in range(4)]

            sourceData = np.ndarray(bandShape,dtype='float64') + np.nan # create unique instance to avoid MemoryError
            for i in range(nbands_key):
                brtoa_name = key + "_" + str(i + 1)
                # print(brtoa_name)
                band = balticPACProduct.addBand(brtoa_name, ProductData.TYPE_FLOAT64)
                #if outputProductFormat == 'BEAM-DIMAP':
                ProductUtils.copySpectralBandProperties(bsources[i], band)
                band.setNoDataValue(np.nan)
                band.setNoDataValueUsed(True)

                sourceData[sline:eline+1,scol:ecol+1] = data[:, i].reshape(bandShape_subset)
                band.setRasterData(ProductData.createInstance(sourceData))


    # Create empty bands for scalar fields
    if not scalar_dict is None:
        sourceData = np.ndarray(bandShape,dtype='float64') + np.nan # create unique instance to avoid MemoryError
        for key in scalar_dict.keys():
            singleBand = balticPACProduct.addBand(key, ProductData.TYPE_FLOAT64)
            singleBand.setNoDataValue(np.nan)
            singleBand.setNoDataValueUsed(True)
            data = scalar_dict[key].get('data')
            if not data is None:
                sourceData[sline:eline+1,scol:ecol+1] = data.reshape(bandShape_subset)
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

    if add_Geometry and not copyOriginalProduct:
        oaa, oza, saa, sza = angle_Reader(product, sensor, subset=subset)
        if sensor == 'OLCI':
            geomNames = ['OAA', 'OZA', 'SAA', 'SZA']
        elif sensor == 'S2MSI':
            geomNames = ['view_azimuth_mean', 'view_zenith_mean', 'sun_azimuth', 'sun_zenith']

        dataList = [oaa, oza, saa, sza]

        sourceData = np.ndarray(bandShape,dtype='float64') + np.nan # create unique instance to avoid MemoryError
        for gn, data in zip(geomNames, dataList):
            singleBand = balticPACProduct.addBand(gn, ProductData.TYPE_FLOAT64)
            singleBand.setNoDataValue(np.nan)
            singleBand.setNoDataValueUsed(True)

            sourceData[sline:eline+1,scol:ecol+1] = data.reshape(bandShape_subset)
            singleBand.setRasterData(ProductData.createInstance(sourceData))


    return balticPACProduct







def polymer_matrix(bands_sat,bands,valid,rho_g,rho_r,sza,oza,wavelength):
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

def check_and_constrain_iop(log_iop, inputRange):
    for i in range(log_iop.shape[1]):
        var = list(inputRange.keys())[i] # IOPs are first variables
        mi = inputRange[var][0]
        ma = inputRange[var][1]
        log_iop[:, i][log_iop[:, i] < mi] = mi
        log_iop[:, i][log_iop[:, i] > ma] = ma
    return log_iop


def baltic_AC(scene_path='', filename='', outpath='', sensor='', subset=None, addName = '', outputSpectral=None,
              outputScalar=None, correction='HYGEOS', copyOriginalProduct=False, outputProductFormat="BEAM-DIMAP",
              atmosphericAuxDataPath = None, niop=5, add_Idepix_Flags=True, add_L2Flags=False, add_c2rccIOPs=False, NNversion='v3'):
    """
    Main function to run the Baltic+ AC based on forward NN
    correction: 'HYGEOS' or 'IPF' for Rayleigh+glint correction
    """


    # Read the NNs
    read_NNs(sensor, NNversion)

    # Get sensor & AC bands
    global bands_sat, bands_rw, bands_corr, bands_chi2, bands_forwardNN, bands_backwardNN, bands_normNN, bands_abs
    global iband_corr, iband_chi2, iband_forwardNN, iband_backwardNN, iband_abs
    bands_sat, bands_rw, bands_corr, bands_chi2, bands_forwardNN, bands_backwardNN, bands_normNN, bands_abs = get_bands.main(sensor,"dummy")
    nbands = len(bands_sat)
    iband_corr = np.searchsorted(bands_sat, bands_corr)
    iband_chi2 = np.searchsorted(bands_sat, bands_chi2)
    iband_forwardNN = np.searchsorted(bands_sat, bands_forwardNN)
    iband_backwardNN = np.searchsorted(bands_sat, bands_backwardNN)
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
    if subset is None:
        height = product.getSceneRasterHeight()
        width = product.getSceneRasterWidth()
    else:
        sline,eline,scol,ecol = subset
        height = eline - sline + 1
        width = ecol - scol + 1
    npix = width*height

    # Read L1B product and convert Radiance to reflectance
    rho_toa = radianceToReflectance_Reader(product, sensor=sensor, subset=subset)

    # Classify pixels with Level-1 flags
    valid = check_valid_pixel_expression_L1(product, sensor, subset=subset)
    print("%d valid pixels on %d"%(np.sum(valid), len(valid)))

    if add_Idepix_Flags:
        idepixProduct = run_IdePix_processor(product, sensor)
        validIdepix = check_valid_pixel_expression_Idepix(idepixProduct, sensor, subset=subset)
        valid = np.logical_and(valid, validIdepix)
    else:
        idepixProduct=None

    # Read geometry and compute relative azimuth angle
    oaa, oza, saa, sza = angle_Reader(product, sensor, subset=subset)
    raa, nn_raa = calculate_diff_azim(oaa, saa)

    # Read wavelength (per-pixel for OLCI) and geolocation
    if sensor == 'OLCI':
        # Read per-pixel wavelength
        wavelength = Level1_Reader(product, sensor, 'lambda0', reshape=False, subset=subset)
        # Read latitude, longitude
        latitude = get_band_or_tiePointGrid(product, 'latitude', reshape=False, subset=subset)
        longitude = get_band_or_tiePointGrid(product, 'longitude', reshape=False, subset=subset)
    elif sensor == 'S2MSI':
        # Duplicate wavelengths for all pixels
        wavelength = np.tile(bands_sat,(len(sza),1)) #TODO: integrate band with S2 SRF
        # Read latitude, longitude
        latitude, longitude = getGeoPositionsForS2Product(product)

    # Read day in the year
    yday = get_yday(product, reshape=False, subset=subset)

    # Read meteo data
    print( "Read meteo data")
    if sensor == 'OLCI':
        pressure = get_band_or_tiePointGrid(product, 'sea_level_pressure', reshape=False, subset=subset)
        ozone = get_band_or_tiePointGrid(product, 'total_ozone', reshape=False, subset=subset)
        tcwv = get_band_or_tiePointGrid(product, 'total_columnar_water_vapour', reshape=False, subset=subset)
        wind_u = get_band_or_tiePointGrid(product, 'horizontal_wind_vector_1', reshape=False, subset=subset)
        wind_v = get_band_or_tiePointGrid(product, 'horizontal_wind_vector_2', reshape=False, subset=subset)
        windm = np.sqrt(wind_u*wind_u+wind_v*wind_v)
        altitude = get_band_or_tiePointGrid(product, 'altitude', reshape=False, subset=subset)
    elif sensor == 'S2MSI':
        bandNames = list(product.getBandNames())
        ## only for match-up data with included meteorology.
        if ('pressure' in bandNames) and ('ozone' in bandNames) and ('tcwv' in bandNames) \
                and ('wind_u' in bandNames) and ('wind_v' in bandNames):
            pressure = get_band_or_tiePointGrid(product, 'pressure', reshape=False, subset=subset)
            ozone = get_band_or_tiePointGrid(product, 'ozone', reshape=False, subset=subset)
            tcwv = get_band_or_tiePointGrid(product, 'tcwv', reshape=False, subset=subset)
            wind_u = get_band_or_tiePointGrid(product, 'wind_u', reshape=False, subset=subset)
            wind_v = get_band_or_tiePointGrid(product, 'wind_v', reshape=False, subset=subset)
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
    Aatm, Aatm_inv = polymer_matrix(bands_sat, bands_corr, valid, rho_g, rho_r, sza, oza, wavelength)

    # Core AC
    print("Inversion")
    rho_w, rho_wmod, log_iop, rho_ag, rho_ag_mod, l2flags, chi2 = AC_backward(rho_rc, td, wavelength, sza, oza, nn_raa, valid, niop, Aatm, Aatm_inv)
    print("")

    # Set absorption band to NaN
    rho_ag[:,iband_abs] = np.NaN
    rho_ag_mod[:,iband_abs] = np.NaN
    rho_rc[:,iband_abs] = np.NaN
    rho_w[:,iband_abs] = np.NaN

    # Apply normalisation
    print("Normalize spectra")
    angle0 = np.zeros(npix)
    rho_wn = np.zeros((npix, nbands)) + np.nan
    rho_wn[:,iband_forwardNN] = apply_forwardNN(log_iop, angle0, angle0, angle0, valid)

    #l2flags[np.array(oorFlagArray==1)] += 2**1 TODO flags?

    # TODO uncertainties
    # unc_rhow =

    ###
    # Writing a product
    # input: spectral_dict holds the spectral fields
    #        scalar_dict holds the scalar fields
    ###
    print("Write output")
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
        log_iop2 = apply_backwardNN(rho_w[:,iband_backwardNN], sza, oza, nn_raa, valid)
        iop_names = ['apig', 'adet', 'a_gelb', 'bpart', 'bwit']
        if scalar_dict is None:
            scalar_dict = {}
        for i, field in enumerate(iop_names):
            scalar_dict[field] = {'data': np.exp(log_iop2[:,i])}

    write_BalticP_AC_Product(product, baltic__product_path, sensor, spectral_dict, scalar_dict,
                             copyOriginalProduct, outputProductFormat, addName,
                             add_Idepix_Flags=add_Idepix_Flags, idepixProduct=idepixProduct,
                             add_L2Flags=add_L2Flags, L2FlagArray=l2flags,
                             add_Geometry=True,subset=subset)

    product.closeProductReader()

    # Close TF session
    if session is not None:
        session.close()

def Rmod_MSA(wav, rho_ag, alpha, lambda_l, wav0=865):
    '''
    MSA analytical aerosol reflectance model
    rho_a in [-0.03, 0.15]
    alpha in [-3., 0.2]
    lamnbda_l in [300, 1400]
    '''
    k = -0.5*(wav0/lambda_l)**alpha
    pow_alpha = (wav/wav0)**alpha

    return rho_ag * pow_alpha * (1 + k*pow_alpha)/(1 + k)

def Jac_MSA(wav, rho_ag, alpha, lambda_l, wav0=865):
    """ Jacobian of Rmod_MSA"""
    pow_alpha = (wav/wav0)**alpha
    k = -0.5*(wav0/lambda_l)**alpha
    f_lambda = -0.5*(wav/lambda_l)**alpha

    j1 = pow_alpha*(1+ k *pow_alpha)/(1+k)
    j2 = rho_ag * pow_alpha * ( np.log(wav/wav0)*(1+k*pow_alpha)/(1+k) +
                                np.log(wav/lambda_l)*f_lambda/(1+k) -
                                np.log(wav0/lambda_l)*k/(1+k)*(1+f_lambda)/(1+k))

    j3 = rho_ag * pow_alpha * alpha/lambda_l *(-f_lambda*(1+k) + (1+f_lambda)*k)/((1+k)*(1+k))

    return np.transpose(np.array([j1, j2, j3]))

def Jac_MSA_LS(x, wav, Rprime, wav0=865):
    return Jac_MSA(wav, *x, wav0=wav0)

def residual_MSA(x, wav, Rprime, wav0=865):
    return (Rmod_MSA(wav, *x, wav0=wav0) - Rprime)

def spectral_MSA(wav, Rprime, x0=[0.0, -1., 500.], wav0=865):
    '''
    Spectral fit with MSA model
    wav in nm
    '''

    #res, pcov = curve_fit(Rmod_MSA, wav, Rprime, kwargs={'wav0':wav0}, p0=x0, bounds=([-0.03,-3.,300.], [0.15, 0.2, 1400.]), jac=Jac_MSA, ftol=1e-3)

    #res_min = least_squares(residual_MSA, x0, args=(wav, Rprime), kwargs={'wav0':wav0}, jac=Jac_MSA_LS, ftol=1e-3, bounds=([-0.03,-3.,300.], [0.15, 0.2, 1400.]))
    res_min = least_squares(residual_MSA, x0, args=(wav, Rprime), kwargs={'wav0':wav0}, jac=Jac_MSA_LS, ftol=1e-3, bounds=([-0.1,-3.,300.], [0.15, 0.2, 1400.]), max_nfev=3)
    res = res_min.x

    return res

def AC_backward(rho_rc, td, wavelength, sza, oza, nn_raa, valid, niop, Aatm, Aatm_inv):
    global n_evaluation
    n_evaluation = 0

    # Reference band
    wav_ref = float(bands_forwardNN[-3])
    iband_ref = iband_forwardNN[-3]

    # Define dimension
    n_dim = 3
    n_vertex = n_dim + 1
    nbands = len(bands_sat)
    n_pix  = np.count_nonzero(valid) # Apply NM only on valid pixel
    range_pix = np.arange(n_pix)

    iter_max = 10#20
    n_iter_NM_max = 20#20
    n_iter_NM = 0
    simplex = np.ndarray((n_pix, n_vertex, n_dim))
    chi2 = np.ndarray((n_pix, n_vertex))

    # Start NM - Define first simplex[n_pix, n_vertex, n_dim]
    #x0 = np.zeros(3)
    #xbest = np.tile(x0,(n_pix,1))
    rho_ag = np.zeros((n_pix, nbands)) + np.NaN
    for ipix in range(n_pix):
        rho_ag[ipix,:] = Rmod_MSA(wavelength[valid][ipix], rho_rc[valid][ipix,iband_ref]*0.5, -1, 500., wav0=wav_ref)
    xbest = np.einsum('...ij,...j->...i', Aatm_inv[valid], rho_ag[:, iband_corr])

    while n_iter_NM < n_iter_NM_max:
        print ("#### Iter NM %d")%n_iter_NM

        # Update first simplex during overall NM iteration
        if n_iter_NM >=0:
            h = 0.02
            for iv in range(n_vertex):
                simplex[:, iv, :] = xbest[:, :]
                if iv >= 1:
                    simplex[:, iv, iv-1] *= 1. + h

        # Evaluate function at first simplex
        for iv in range(n_vertex):
            chi2[:,iv], tmp, tmp, tmp, tmp, tmp = evaluate_chi2(simplex[:,iv], rho_rc, td, wavelength, wav_ref, iband_ref, sza, oza, nn_raa, valid, Aatm_inv, Aatm)

        # Loop
        n_iter = 0
        n_reflection = np.zeros(n_pix)
        n_expansion = np.zeros(n_pix)
        n_contraction =np.zeros(n_pix)
        n_reduction = np.zeros(n_pix)
        i_reduction = np.zeros(n_pix, dtype=bool)
        diam = np.zeros(n_pix)
        yr = np.zeros(n_pix)
        ynew = np.zeros(n_pix)
        while n_iter<iter_max:

            print ("Iter %d")%n_iter
            # Init action
            i_reduction[:] = False

            # Compute size of simplex (size of diam is n_pix)
            diam = diameter(simplex)
            if max(diam)< 1E-4:
                print ("Diameter=%f -- break")%max(diam)
                break

            # Order vertices
            order = np.argsort(chi2, axis=1)
            ibest = order[:,0] # best
            isecond = order[:,-2] # second to worse
            iworse = order[:,-1] # worse
            xbest = simplex[range_pix,ibest]
            xworse = simplex[range_pix,iworse]
            xnew = simplex[range_pix,ibest] # Values re-evaluated for expansion or contraction, but need realistic dummy for other
            ybest = chi2[range_pix,ibest]
            ysecond = chi2[range_pix,isecond]
            yworse = chi2[range_pix,iworse]

            # Check convergence
            if max(ybest) < 1.E-6:
                print ("Max(ybest)=%f -- break")%max(ybest)
                break

            # Compute centroid
            ikeep = order[:,:-1]
            range_keep = np.tile(range_pix,(n_vertex-1,1)).transpose()
            xc = simplex[range_keep,ikeep].mean(1)

            # Reflect worse vertex
            xr = xc + (xc-xworse)
            # Evaluate reflection
            yr, tmp, tmp, tmp, tmp, tmp = evaluate_chi2(xr, rho_rc, td, wavelength, wav_ref, iband_ref, sza, oza, nn_raa, valid, Aatm_inv, Aatm)

            # Replace vertex
            i_reflection = (ybest <= yr) & (yr <  ysecond)
            if i_reflection.any():
                simplex[range_pix[i_reflection],iworse[i_reflection]] = xr[i_reflection]
                chi2[range_pix[i_reflection],iworse[i_reflection]] = yr[i_reflection]
                n_reflection[i_reflection] += 1

            # Dectect expansion
            i_expansion = yr < ybest
            if i_expansion.any():
                xe = xc + 2.*(xc-xworse)
                xnew[i_expansion] = xe[i_expansion]

            # Detect contraction
            i_contraction = yr >= ysecond
            if i_contraction.any():
                i_ce = i_contraction & (yr < yworse)
                if i_ce.any():
                    xce = xc + 0.5*(xc-xworse)
                    xnew[i_ce] = xce[i_ce]
                i_ci = i_contraction & (yr >= yworse)
                if i_ci.any():
                    xci = xc - 0.5*(xc-xworse)
                    xnew[i_ci] = xci[i_ci]

            # Evaluate new vertex for expansion or contraction
            if i_expansion.any() or i_contraction.any():
                ynew, tmp, tmp, tmp, tmp, tmp = evaluate_chi2(xnew, rho_rc, td, wavelength, wav_ref, iband_ref, sza, oza, nn_raa, valid, Aatm_inv, Aatm)

            # Apply expansion
            if i_expansion.any():
                i_e = i_expansion & (ynew <= yr)
                if i_e.any():
                    simplex[range_pix[i_e],iworse[i_e]] = xnew[i_e]
                    chi2[range_pix[i_e],iworse[i_e]] = ynew[i_e]
                i_e = i_expansion & (ynew > yr)
                if i_e.any():
                    simplex[range_pix[i_e],iworse[i_e]] = xr[i_e]
                    chi2[range_pix[i_e],iworse[i_e]] = yr[i_e]
                n_expansion[i_expansion] += 1

                # Apply contraction
            if i_contraction.any():
                if i_ce.any():
                    ii = i_ce & (ynew <= yr)
                    if ii.any():
                        simplex[range_pix[ii],iworse[ii]] = xnew[ii]
                        chi2[range_pix[ii],iworse[ii]] = ynew[ii]
                        n_contraction[ii] += 1
                    ii = i_ce & (ynew > yr)
                    if ii.any():
                        i_reduction[ii] = True
                if i_ci.any():
                    ii = i_ci & (ynew < yworse)
                    if ii.any():
                        simplex[range_pix[ii],iworse[ii]] = xnew[ii]
                        chi2[range_pix[ii],iworse[ii]] = ynew[ii]
                        n_contraction[ii] += 1
                    ii = i_ci & (ynew >= yworse)
                    if ii.any():
                        i_reduction[ii] = True

            # Reduction (shrink)
            if i_reduction.any():
                #print "reduction:",np.count_nonzero(i_reduction)
                for iv in range(n_vertex):
                    simplex[i_reduction,iv] = simplex[i_reduction,ibest[i_reduction]] + 0.5*(simplex[i_reduction,iv]-simplex[i_reduction,ibest[i_reduction]])
                # Evaluate reduction
                for iv in range(n_vertex):
                    y, tmp, tmp, tmp, tmp, tmp = evaluate_chi2(simplex[:,iv], rho_rc, td, wavelength, wav_ref, iband_ref, sza, oza, nn_raa, valid, Aatm_inv, Aatm)
                    chi2[range_pix[i_reduction], iv] = y[i_reduction] # Care y defined only for dopix=i_reduction
                n_reduction[i_reduction] += 1

            n_iter = n_iter + 1

        # End of algorithm
        print ("Niter=",n_iter)
        print ("Reflection=",n_reflection[0:10], np.count_nonzero(n_reflection))
        print ("Expansion=",n_expansion[0:10], np.count_nonzero(n_expansion))
        print ("Contraction=",n_contraction[0:10],  np.count_nonzero(n_contraction))
        print ("Reduction=",n_reduction[0:10], np.count_nonzero(n_reduction))
        print ("Evaluation=",n_evaluation)

        # Identify best vertex
        print ("Identify best vertex")
        order = np.argsort(chi2,axis=1)
        ibest = order[:,0] # best
        xbest = simplex[range_pix,ibest]

        n_iter_NM = n_iter_NM + 1

    # Final evaluation at best vertex
    print ("Evaluate best vertex")
    n_pix_all = oza.shape[0]
    chi2 = np.zeros(n_pix_all) + np.NaN
    rho_w = np.zeros((n_pix_all, nbands)) + np.NaN
    rho_wmod = np.zeros((n_pix_all, nbands)) + np.NaN
    rho_ag = np.zeros((n_pix_all, nbands)) + np.NaN
    rho_ag_mod = np.zeros((n_pix_all, nbands)) + np.NaN
    log_iop = np.zeros((n_pix_all, niop)) + np.NaN
    chi2[valid], rho_w[valid], rho_wmod[valid], rho_ag[valid], rho_ag_mod[valid], log_iop[valid] = evaluate_chi2(xbest, rho_rc, td, wavelength, wav_ref, iband_ref, sza, oza, nn_raa, valid, Aatm_inv, Aatm)

    l2flags = np.zeros(n_pix_all)
    # TODO define success =
    #if not success:
    #    l2flags[ipix] += 2 ** 2

    return rho_w, rho_wmod, log_iop, rho_ag, rho_ag_mod, l2flags, chi2

def diameter(simplex):

    n_pix, n_vertex, n_dim = simplex.shape
    d = np.zeros(n_pix)
    for i in range(n_vertex):
        for j in range(i+1,n_vertex):
            dij = np.linalg.norm(simplex[:,i]-simplex[:,j],axis=1)
            d = np.amax((d,dij),axis=0)
    return d

def evaluate_chi2(vertices, rho_rc, td, wavelength, wav_ref, iband_ref, sza, oza, nn_raa, valid, Aatm_inv, Aatm, dopix=np.array(False)):
    global n_evaluation
    n_evaluation += 1

    nbands = len(bands_sat)

    # Limit data to dopix
    if dopix.any():
        n_pix = np.count_nonzero(dopix)
        index = dopix
    else:
        n_pix = vertices.shape[0]
        index = range(n_pix)
    #all_valid = np.ones(n_pix, dtype=bool)

    # Compute rho_ag_mod
    rho_ag_mod = np.zeros((n_pix, nbands))# + np.NaN
    #for ipix in range(n_pix):
    #    rho_ag_mod[ipix,:] = Rmod_MSA(wavelength[valid][index][ipix], *(vertices[index][ipix]), wav0=wav_ref)
    rho_ag_mod[:, :] = np.einsum('...ij,...j->...i', Aatm[valid][index], vertices[index])

    # Fit rhow_mod
    rho_wmod = np.zeros((n_pix, nbands)) + np.NaN
    rho_w = (rho_rc[valid][index] - rho_ag_mod)/td[valid][index]
    log_iop = apply_backwardNN(rho_w[:,iband_backwardNN], sza[valid][index], oza[valid][index], nn_raa[valid][index], valid[valid][index])
    # Check iop range and apply constraints to forwardNN input range
    #log_iop = check_and_constrain_iop(log_iop, inputRange_forward) # TODO to be applied after backwardNN ?
    rho_wmod[:,iband_forwardNN] = apply_forwardNN(log_iop, sza[valid][index], oza[valid][index], nn_raa[valid][index], valid[valid][index])

    # Evaluate rho_ag (not used)
    rho_ag = rho_rc[valid][index] - td[valid][index]*rho_wmod

    # Compute chi2
    chi2 = np.sum((rho_wmod[:,iband_chi2]-rho_w[:,iband_chi2])**2, axis=1)

    chi2[np.any(np.isnan(log_iop),axis=1)] = 100.

    return chi2, rho_w, rho_wmod, rho_ag, rho_ag_mod, log_iop