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
sys.path.append("C:\\Users\Dagmar\.snap\snap-python")
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
from misc import default_ADF, nlinear, scale_minmax
import luts_olci, luts_msi
import lut_hygeos
from auxdata_handling import setAuxData, checkAuxDataAvailablity, getGeoPositionsForS2Product, yearAndDoyAndHourUTC

# Set locale for proper time reading with datetime
# locale.setlocale(locale.LC_ALL, 'en_US.UTF_8')

# Define NNs as global variables
def read_NNs(sensor, NNversion, NNIOPversion):
    global nn_forward, nn_backward, nn_backward_iop
    global nnForward_IO, nnBackward_IO, nnBackwardIOP_IO 
    global session

    # Define paths of NNs
    filePath = glob.glob(os.path.join('NNs',sensor,NNversion,'forward','*.h5'))
    if filePath == []:
        print("Missing NN in %s format in NNs/%s/%s/forward/"%('h5',sensor,NNversion))
        sys.exit(1)
    else:
        nnForwardFilePath = filePath[0]

    filePath = glob.glob(os.path.join('NNs',sensor,NNversion,'backward','*.h5'))
    if filePath == []:
        print("Missing NN in %s format in NNs/%s/%s/backward/"%('h5',sensor,NNversion))
        sys.exit(1)
    else:
        nnBackwardFilePath = filePath[0]

    filePath = glob.glob(os.path.join('NNs',sensor,NNIOPversion,'backward','*.h5'))
    if filePath == []:
        print("Missing NN in %s format in NNs/%s/%s/backward/"%('h5',sensor,NNIOPversion))
        sys.exit(1)
    else:
        nnBackwardIOPFilePath = filePath[0]

    # Open NNs
    nn_forward = open_NN(nnForwardFilePath, NNversion)
    nn_backward = open_NN(nnBackwardFilePath, NNversion)
    nn_backward_iop = open_NN(nnBackwardIOPFilePath, NNIOPversion)

    # Define paths of .net files containing the ranges
    filePath = glob.glob(os.path.join('NNs',sensor,NNversion,'forward','*.net'))
    if filePath == []:
        print("Missing NN in %s format in NNs/%s/%s/forward/"%('net',sensor,NNversion))
        sys.exit(1)
    else:
        nnForwardFilePathNet = filePath[0]

    filePath = glob.glob(os.path.join('NNs',sensor,NNversion,'backward','*.net'))
    if filePath == []:
        print("Missing NN in %s format in NNs/%s/%s/backward/"%('net',sensor,NNversion))
        sys.exit(1)
    else:
        nnBackwardFilePathNet = filePath[0]

    filePath = glob.glob(os.path.join('NNs',sensor,NNIOPversion,'backward','*.net'))
    if filePath == []:
        print("Missing NN in %s format in NNs/%s/%s/backward/"%('net',sensor,NNIOPversion))
        sys.exit(1)
    else:
        nnBackwardIOPFilePathNet = filePath[0]

    # Define ranges for forward NN and backward NN
    nnForward_IO = read_NN_IO_fromFile(nnForwardFilePathNet)
    nnBackward_IO = read_NN_IO_fromFile(nnBackwardFilePathNet)
    nnBackwardIOP_IO = read_NN_IO_fromFile(nnBackwardIOPFilePathNet)
 
def open_NN(nnFilePath, NNversion):
    global session

    # Open session for tf v1.X
    TF_version = tf.__version__.split('.')[0]
    if TF_version == '1':
        if 'session' not in globals() or (session is None):
            session = tf.InteractiveSession()
    else:
        session = None
    # Read NNs
    nn_object = tf.keras.models.load_model(nnFilePath)

    return nn_object

def read_NN_IO_fromFile(nnFilePath):
    """ Read input range in the .net file """
    file = open(nnFilePath, 'r')
    lines = [line.rstrip('\n') for line in file]
    file.close()
    Ninput = 0
    Noutput = 0
    for line in lines:
        if 'the net has' in line:
            if 'input' in line:
                Ninput = int(line.split(' ')[3])
            if 'output' in line:
                Noutput = int(line.split(' ')[3])

    inputVariables = []
    inputRange = np.zeros((Ninput, 2))
    outputRange = np.zeros((Noutput, 2))
    bands = []  # Either input or output bands

    iIN = 0
    iOUT = 0
    scaling = False
    for line in lines:
        if line.startswith('input'):
            inputVariables.append(line.split('is ')[1].split(' ')[0])
            t = line.split('[')[1][:-1]
            t = t.split(',')
            inputRange[iIN, 0] = float(t[0])
            inputRange[iIN, 1] = float(t[1])
            if 'log_rw_' in line:
                bands.append(int(line.split('log_rw_')[1].split(' ')[0]))
            iIN += 1
        if line.startswith('output'):
            t = line.split('[')[1][:-1]
            outputRange[iOUT, 0] = float(t.split(',')[0])
            outputRange[iOUT, 1] = float(t.split(',')[1])
            if 'log_rw_' in line:
                bands.append(int(line.split('log_rw_')[1].split(' ')[0]))
            iOUT += 1
        if line.startswith('scaling'):
            scaling = line.split('is ')[1].split(' ')[0].lower() == 'true'

    nnIO =  {
            'inputVariables': inputVariables,
            'inputRange': inputRange,
            'outputRange': outputRange,
            'bands': bands,
            'scaling': scaling
            }
   
    return collections.namedtuple('NN_IO', nnIO.keys())(*nnIO.values())


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
    if product.getProductType() == 'CSV':
        idepixParameters.put("computeCloudBuffer", 'false')
        idepixParameters.put("computeCloudShadow", 'false')
    else:
        idepixParameters.put("computeCloudBuffer", 'true')
        idepixParameters.put("cloudBufferWidth", '2')

    idepixProducts = HashMap()
    # idepixProducts.put("l1bProduct", product) # SNAP v6 ?
    idepixProducts.put("sourceProduct", product)

    idepix_product = None

    if sensor == 'OLCI':
        #idepix_product = GPF.createProduct("Idepix.Sentinel3.Olci", idepixParameters, product) # SNAP v6
        idepix_product = GPF.createProduct("Idepix.Olci", idepixParameters, idepixProducts) # SNAP v7
    elif sensor == 'S2MSI':
        idepixParameters.put("computeCloudBufferForCloudAmbiguous", 'true')
        idepix_product = GPF.createProduct("Idepix.S2", idepixParameters, idepixProducts) # SNAP v7

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

def apply_forwardNN_TF(log_iop, sun_zenith, view_zenith, diff_azimuth, valid):
    """
    Apply the forwardNN: IOP to rhow
    input: numpy array log_iop, shape = (Npixels x log_iops= (log_apig, log_adet, log a_gelb, log_bpart, log_bwit)),
            np.array sza, shape = (Npixels,)
            np.array oza, shape = (Npixels,)
            np.array raa, shape = (Npixels,); range: 0-180
    returns: np.array rhow, shape = (Npixels, wavelengths)
    """

    # Initialise output
    nBands = len(iband_forwardNN)
    rhow = np.zeros((log_iop.shape[0], nBands)) + np.NaN

    # Initialise input
    nInput = 3 + 5 # angle + IOPs
    if 'temperature' in nnForward_IO.inputVariables:
        nInput +=1
    if 'salinity' in nnForward_IO.inputVariables:
        nInput += 1
    NN_input = np.zeros((log_iop.shape[0], nInput)) -1. # CARE -1 used by default for IOP when niop < 5

    # Prepare the NN
    if 'cos_' in nnForward_IO.inputVariables[0]:
        NN_input[:,0] = np.cos(sun_zenith * np.pi / 180.)
        NN_input[:,1] = np.cos(view_zenith * np.pi / 180.)
        NN_input[:,2] = np.cos(diff_azimuth * np.pi / 180.)
    else:
        NN_input[:, 0] = sun_zenith
        NN_input[:, 1] = view_zenith
        NN_input[:, 2] = diff_azimuth
    i_input = 3
    if 'temperature' in nnForward_IO.inputVariables:
        NN_input[:,i_input] = 15. # default temperature
        i_input += 1
    if 'salinity' in nnForward_IO.inputVariables:
        NN_input[:,i_input] = 35. # default salinity
        i_input += 1
    for i in range(log_iop.shape[1]):
        NN_input[:,i_input+i] = log_iop[:,i]

    # Scale inputs if necessary, first constraining in the range
    if nnForward_IO.scaling:
        NN_input[:,i_input:] = check_and_constrain_iop(NN_input[:,i_input:], nnForward_IO)
        NN_input = scale_minmax(NN_input, nnForward_IO.inputRange)

    # Limit to valid pixels
    NN_input = np.array(NN_input[valid, :], dtype='float32')

    # Launch NN
    if session is not None:
        log_rhow = nn_forward(NN_input).eval(session=session)
    else:
        log_rhow = nn_forward(NN_input)

    # Scale outputs if necessary
    if nnForward_IO.scaling:
        log_rhow = scale_minmax(np.array(log_rhow), nnForward_IO.outputRange, reverse=True)

    # Get NN output
    for i in range(nBands):
        rhow[valid,i] = np.exp(log_rhow[:,i])

    return rhow

def apply_backwardNN_TF(rhow, sun_zenith, view_zenith, diff_azimuth, valid, NN_IO, NNIOP=False):
    """
    Apply the backwardNN: rhow to IOP
    input: numpy array rhow, shape = (Npixels x rhow = (log_rw_band1, log_rw_band2_.... )),
            np.array sza, shape = (Npixels,)
            np.array oza, shape = (Npixels,)
            np.array raa, shape = (Npixels,); range: 0-180
    returns: np.array log_iop, shape = (Npixels x log_iop= (log_apig, log_adet, log a_gelb, log_bpart, log_bwit))

    T, S: currently constant #TODO take ECMWF temperature at sea surface?
    """

    # Initialise output
    log_iop = np.zeros((rhow.shape[0], 5))
    log_iop[:] = NN_IO.outputRange[:,0] # set default values, for invalid NN inversion (rhow<0)

    # Initialise input
    rhow_pos = np.all(rhow>0, axis=1) # limit to rhow >0
    valid2 = valid & rhow_pos
    nvalid = np.count_nonzero(valid2)

    nInput = 3 + rhow.shape[1] # angle + log_rw
    if 'temperature' in NN_IO.inputVariables:
        nInput +=1
    if 'salinity' in NN_IO.inputVariables:
        nInput += 1
    NN_input = np.zeros((nvalid, nInput), dtype='float32') 

    # Prepare the NN
    if 'cos_' in NN_IO.inputVariables[0]:
        NN_input[:,0] = np.cos(sun_zenith[valid2] * np.pi / 180.)
        NN_input[:,1] = np.cos(view_zenith[valid2] * np.pi / 180.)
        NN_input[:,2] = np.cos(diff_azimuth[valid2] * np.pi / 180.)
    else:
        NN_input[:, 0] = sun_zenith[valid2]
        NN_input[:, 1] = view_zenith[valid2]
        NN_input[:, 2] = diff_azimuth[valid2]
    i_input = 3
    if 'temperature' in NN_IO.inputVariables:
        NN_input[:,i_input] = 15. # default temperature
        i_input += 1
    if 'salinity' in NN_IO.inputVariables:
        NN_input[:,i_input] = 35. # default salinity
        i_input += 1
    NN_input[:,i_input:] = np.log(rhow[valid2,:])
    
    # Scale inputs if necessary, , first constraining log_rw in the range
    if NN_IO.scaling:
        NN_input[:,i_input:] = check_and_constrain_rw(NN_input[:,i_input:], NN_IO)
        NN_input = scale_minmax(NN_input, NN_IO.inputRange)

    # Launch the NN
    if NNIOP:
        if session is not None:
            log_iop[valid2,:] = nn_backward_iop(NN_input).eval(session=session)
        else:
            log_iop[valid2,:] = nn_backward_iop(NN_input)
    else:
        if session is not None:
            log_iop[valid2,:] = nn_backward(NN_input).eval(session=session)
        else:
            log_iop[valid2,:] = nn_backward(NN_input)

    # Scale outputs if necessary
    if NN_IO.scaling:
        log_iop[valid2,:] = scale_minmax(log_iop[valid2,:], NN_IO.outputRange, reverse=True)

    return log_iop

def write_BalticP_AC_Product(product, baltic__product_path, sensor, spectral_dict, scalar_dict=None,
                             copyOriginalProduct=False, outputProductFormat="BEAM-DIMAP", addname='',
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

    dirname = os.path.dirname(baltic__product_path)
    outname, ext = os.path.splitext(os.path.basename(baltic__product_path))
    if outputProductFormat == "BEAM-DIMAP":
        baltic__product_path = os.path.join(dirname, outname + addname +'.dim')
    elif outputProductFormat == 'CSV':
        baltic__product_path = os.path.join(dirname, outname + addname +'.csv')

    balticPACProduct = Product('balticPAC', 'balticPAC', width, height)
    balticPACProduct.setFileLocation(File(baltic__product_path))

    # Copy geocoding
    ProductUtils.copyGeoCoding(product, balticPACProduct)
    # replacement by Tonio below does not work, geocoding is on the full image, not subset
    """# PixelSubsetRegion = jpy.get_type('org.esa.snap.core.subset.PixelSubsetRegion')
    ProductSubsetDef = jpy.get_type('org.esa.snap.core.dataio.ProductSubsetDef')
    # subset_region = PixelSubsetRegion(scol, sline, ecol, eline)
    subset_def = ProductSubsetDef()
    subset_def.setRegion(scol, sline, width, height)
    product.transferGeoCodingTo(balticPACProduct, subset_def)"""

    # writer = ProductIO.getProductWriter(outputProductFormat)
    # writer.writeProductNodes(balticPACProduct, baltic__product_path)

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

            sourceData = np.ndarray(bandShape,dtype='float32') + np.nan # create unique instance to avoid MemoryError
            for i in range(nbands_key):
                brtoa_name = key + "_" + str(i + 1)
                # print(brtoa_name)
                band = balticPACProduct.addBand(brtoa_name, ProductData.TYPE_FLOAT32)
                if outputProductFormat == 'BEAM-DIMAP':
                    ProductUtils.copySpectralBandProperties(bsources[i], band)
                band.setNoDataValue(np.nan)
                band.setNoDataValueUsed(True)

                sourceData[sline:eline+1,scol:ecol+1] = data[:, i].reshape(bandShape_subset).astype('float32')
                band.setRasterData(ProductData.createInstance(sourceData))


    # Create empty bands for scalar fields
    if not scalar_dict is None:
        sourceData = np.ndarray(bandShape,dtype='float32') + np.nan # create unique instance to avoid MemoryError
        for key in scalar_dict.keys():
            singleBand = balticPACProduct.addBand(key, ProductData.TYPE_FLOAT32)
            singleBand.setNoDataValue(np.nan)
            singleBand.setNoDataValueUsed(True)
            data = scalar_dict[key].get('data')
            if not data is None:
                sourceData[sline:eline+1,scol:ecol+1] = data.reshape(bandShape_subset).astype('float32')
                singleBand.setRasterData(ProductData.createInstance(sourceData))

    if copyOriginalProduct:
        originalBands = product.getBandNames()
        balticBands = balticPACProduct.getBandNames()
        for bb in balticBands:
            originalBands = [ob for ob in originalBands if ob != bb]
        for ob in originalBands:
            singleBand = balticPACProduct.addBand(ob, ProductData.TYPE_FLOAT32)
            singleBand.setNoDataValue(np.nan)
            singleBand.setNoDataValueUsed(True)

            data = get_band_or_tiePointGrid(product,ob)
            sourceData = data.reshape(bandShape).astype('float32')
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

        sourceData = np.ndarray(bandShape,dtype='float32') + np.nan # create unique instance to avoid MemoryError
        for gn, data in zip(geomNames, dataList):
            singleBand = balticPACProduct.addBand(gn, ProductData.TYPE_FLOAT32)
            singleBand.setNoDataValue(np.nan)
            singleBand.setNoDataValueUsed(True)

            sourceData[sline:eline+1,scol:ecol+1] = data.reshape(bandShape_subset)
            singleBand.setRasterData(ProductData.createInstance(sourceData))



    if outputProductFormat == 'BEAM-DIMAP':
        # Set auto grouping
        autoGroupingString = ':'.join(spectral_dict.keys())
        balticPACProduct.setAutoGrouping(autoGroupingString)

    GPF.writeProduct(balticPACProduct, File(baltic__product_path), outputProductFormat, False, ProgressMonitor.NULL)

    balticPACProduct.closeIO()



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

def check_and_constrain_iop(log_iop, NN_IO):
    for i, var in enumerate(['log_apig', 'log_adet', 'log_agelb', 'log_bpart', 'log_bwit']):
        j = NN_IO.inputVariables.index(var)
        mi = NN_IO.inputRange[j,0]
        ma = NN_IO.inputRange[j,1]
        log_iop[:, i][log_iop[:, i] < mi] = mi
        log_iop[:, i][log_iop[:, i] > ma] = ma
    return log_iop

def check_and_constrain_rw(log_rw, NN_IO):
    for i, band in enumerate(NN_IO.bands):
        var = 'log_rw_%d'%band
        j = NN_IO.inputVariables.index(var)
        mi = NN_IO.inputRange[j,0]
        ma = NN_IO.inputRange[j,1]
        log_rw[:, i][log_rw[:, i] < mi] = mi
        log_rw[:, i][log_rw[:, i] > ma] = ma
    return log_rw

def baltic_AC(scene_path='', filename='', outpath='', sensor='', platform='', subset=None, addName = '', outputSpectral=None,
                outputScalar=None, correction='HYGEOS', copyOriginalProduct=False, outputProductFormat="BEAM-DIMAP",
                atmosphericAuxDataPath = None, niop=5, add_Idepix_Flags=True, add_L2Flags=False, add_c2rccIOPs=False,
              runAC=True, NNversion='v1_baltic+', NNIOPversion='c2rcc_20171221'):
    """
    Main function to run the Baltic+ AC based on forward NN
    correction: 'HYGEOS' or 'IPF' for Rayleigh+glint correction
    """


    # Read the NNs
    read_NNs(sensor, NNversion, NNIOPversion)

    # Get sensor & NN bands
    global bands_sat, bands_corr, bands_chi2, bands_forwardNN, bands_backwardNN, bands_abs
    global iband_corr, iband_chi2, iband_forwardNN, iband_backwardNN, iband_backwardNNIOP, iband_abs
    bands_sat, bands_corr, bands_chi2,  bands_abs = get_bands.main(sensor)
    nbands = len(bands_sat)
    # Get band of NNs
    bands_forwardNN = np.array(nnForward_IO.bands) 
    bands_backwardNN = np.array(nnBackward_IO.bands)
    bands_backwardNNIOP = np.array(nnBackwardIOP_IO.bands)
    # Check NNs bands
    for NN, band_set in zip(['forward_NN','backward_NN','backward_NNIOP'], [bands_forwardNN, bands_backwardNN, bands_backwardNNIOP]):
        if not set(band_set).issubset(set(bands_sat)):
            print("Error: bands of %s does not match sensor band:"%NN, band_set)
            sys.exit(1)
    # Check bands_corr and band_chi2 are provided by forward NN. If not, take intersection
    bands_AC = {
            'bands_corr': bands_corr,
            'bands_chi2': bands_chi2
            }
    for band_set, bands_value in zip(bands_AC, bands_AC.values()):
        if not set(bands_value).issubset(set(bands_forwardNN)):
                print("WARNING: %s is not a subset of bands_forwardNN:"%band_set)
                # Searching intersection
                bands_value = list(set(bands_value) & set(bands_forwardNN))
                if len(bands_value) == 0:
                    print("         Intersection empty between %s and bands_forwardNN - Aborted"%band_set)
                    sys.exit(1)
                else:
                    bands_value.sort()
                    bands_AC[band_set] = bands_value
                    print("         %s limited to "%band_set, bands_AC[band_set])
    bands_corr = bands_AC['bands_corr']
    bands_chi2 = bands_AC['bands_chi2']

    # Identify index of bands
    iband_corr = np.searchsorted(bands_sat, bands_corr)
    iband_chi2 = np.searchsorted(bands_sat, bands_chi2)
    iband_forwardNN = np.searchsorted(bands_sat, bands_forwardNN)
    iband_backwardNN = np.searchsorted(bands_sat, bands_backwardNN)
    iband_backwardNNIOP = np.searchsorted(bands_sat, bands_backwardNNIOP)
    iband_abs = np.searchsorted(bands_sat, bands_abs)

    # Initialising a product for Reading with snappy
    product = snp.ProductIO.readProduct(os.path.join(scene_path, filename))

    if sensor=='OLCI' and product.getProductType()=='CSV':
        product.setProductType('OL_1_')

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

    # Read geometry and compute relative azimuth angle
    oaa, oza, saa, sza = angle_Reader(product, sensor, subset=subset)
    raa, nn_raa = calculate_diff_azim(oaa, saa)

    # Classify pixels with Level-1 flags
    valid = check_valid_pixel_expression_L1(product, sensor, subset=subset)
    
    # Check valid geometry (necessary for S2MSI)
    valid[np.isnan(oza)] = False
    valid[np.isnan(sza)] = False
    valid[np.isnan(raa)] = False

    print("%d valid pixels on %d"%(np.sum(valid), len(valid)))

    # Apply Idepix
    if add_Idepix_Flags:
        print("Launch Idepix")
        if product.getBand('quality_flags') is None: #Idepix needs a band of this name to run. L1-flags are evaluated st a different step, so values can be zero here.
            band = product.addBand('quality_flags', ProductData.TYPE_INT32)
            band.setNoDataValue(np.nan)
            band.setNoDataValueUsed(True)
            sourceData = np.zeros((product.getSceneRasterHeight(), product.getSceneRasterWidth()), dtype='uint32') # Idepix launched on full scene, not subset
            band.setRasterData(ProductData.createInstance(sourceData))

        idepixProduct = run_IdePix_processor(product, sensor)

        validIdepix = check_valid_pixel_expression_Idepix(idepixProduct, sensor, subset=subset)
        print('Idepix valid', np.sum(validIdepix))
        valid = np.logical_and(valid, validIdepix)
        print('total valid', np.sum(valid))
    else:
        idepixProduct=None

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
        latitude, longitude = getGeoPositionsForS2Product(product, reshape=False, subset=subset)

    # Read day in the year
    yday = get_yday(product, reshape=False, subset=subset)

    # Read meteo data
    print( "Read meteo data")
    if sensor == 'OLCI':
        pressure = get_band_or_tiePointGrid(product, 'sea_level_pressure', reshape=False, subset=subset)
        ozone = get_band_or_tiePointGrid(product, 'total_ozone', reshape=False, subset=subset) / 2.1415e-5  # convert kg/m2 to DU
        tcwv = get_band_or_tiePointGrid(product, 'total_columnar_water_vapour', reshape=False, subset=subset)
        wind_u = get_band_or_tiePointGrid(product, 'horizontal_wind_vector_1', reshape=False, subset=subset)
        wind_v = get_band_or_tiePointGrid(product, 'horizontal_wind_vector_2', reshape=False, subset=subset)
        windm = np.sqrt(wind_u*wind_u+wind_v*wind_v)
        altitude = get_band_or_tiePointGrid(product, 'altitude', reshape=False, subset=subset)
    elif sensor == 'S2MSI':
        bandNames = list(product.getBandNames())
        ## only for match-up data with included meteorology.
        if ('pressure' in bandNames) and ('ozone' in bandNames) and ('tcwv' in bandNames)\
                and ('wind_u' in bandNames) and ('wind_v' in bandNames):
            pressure = get_band_or_tiePointGrid(product, 'pressure', reshape=False, subset=subset)
            ozone = get_band_or_tiePointGrid(product, 'ozone', reshape=False, subset=subset)
            tcwv = get_band_or_tiePointGrid(product, 'tcwv', reshape=False, subset=subset)
            wind_u = get_band_or_tiePointGrid(product, 'wind_u', reshape=False, subset=subset)
            wind_v = get_band_or_tiePointGrid(product, 'wind_v', reshape=False, subset=subset)
            windm = np.sqrt(wind_v * wind_v + wind_u * wind_u)
            altitude = get_band_or_tiePointGrid(product, 'altitude', reshape=False, subset=subset)
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
                altitude = np.zeros(shape) # zero altitude by defautl #TODO get if from a DEM?
            else:
                print('Please set a path to the AUX data archive.')
                sys.exit(1)

    # Define LUTs
    file_adf_acp = default_ADF[sensor][platform]['file_adf_acp']
    file_adf_ppp = default_ADF[sensor][platform]['file_adf_ppp']
    file_adf_clp = default_ADF[sensor][platform]['file_adf_clp']

    # Read LUTs
    if sensor == 'OLCI':
        adf_acp = luts_olci.LUT_ACP(file_adf_acp)
        adf_ppp = luts_olci.LUT_PPP(file_adf_ppp)
        adf_clp = luts_olci.LUT_CLP(file_adf_clp)
    elif sensor == 'S2MSI':
        # temporary solution: read OLCI ADF and interpolate for wavelength
        adf_acp = luts_olci.LUT_ACP(file_adf_acp)
        adf_ppp = luts_olci.LUT_PPP(file_adf_ppp)
        adf_clp = luts_olci.LUT_CLP(file_adf_clp)
        luts_msi.adjust_S2_luts(adf_acp, adf_ppp)

    if correction == 'HYGEOS':
        LUT_HYGEOS = lut_hygeos.LUT(default_ADF['GENERIC']['file_HYGEOS'])

    file_SRF_wavelength = default_ADF[sensor][platform]['SRF_wavelength']
    file_SRF_weights = default_ADF[sensor][platform]['SRF_weights']

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
        rho_r, rho_molgli, rho_rc, tau_r = Rmolgli_correction_Hygeos(rho_ng, valid, latitude, sza, oza, raa, wavelength,
                                                                     pressure, windm, LUT_HYGEOS, file_SRF_wavelength, file_SRF_weights, altitude)

    if np.sum(valid) != 0 and runAC:
        # Atmospheric model
        print("Compute atmospheric matrices")
        Aatm, Aatm_inv = polymer_matrix(bands_sat, bands_corr, valid, rho_g, rho_r, sza, oza, wavelength)

        # Core AC
        print("Inversion")
        rho_w, rho_wmod, log_iop, rho_ag, rho_ag_mod, l2flags, chi2, unc_rhow = AC_forward(rho_rc, td, wavelength, sza, oza, nn_raa, valid, niop, Aatm, Aatm_inv)
        print("")

        # Set absorption band to NaN
        rho_ag[:,iband_abs] = np.NaN
        rho_ag_mod[:,iband_abs] = np.NaN
        rho_rc[:,iband_abs] = np.NaN
        rho_w[:,iband_abs] = np.NaN
        unc_rhow[:,iband_abs] = np.NaN

        # Apply normalisation
        print("Normalize spectra")
        angle0 = np.zeros(npix)
        rho_wn = np.zeros((npix, nbands)) + np.nan
        rho_wn[:,iband_forwardNN] = apply_forwardNN_TF(log_iop, angle0, angle0, angle0, valid)/rho_wmod[:,iband_forwardNN] * rho_w[:,iband_forwardNN]
    else:
        rho_w = np.zeros((npix, nbands)) + np.nan
        rho_wn = np.zeros((npix, nbands)) + np.nan
        rho_wmod = np.zeros((npix, nbands)) + np.nan
        log_iop = np.zeros((npix, niop)) + np.nan
        rho_ag = np.zeros((npix, nbands)) + np.nan
        rho_ag_mod = np.zeros((npix, nbands)) + np.nan
        l2flags = np.zeros(npix) + np.nan
        chi2 = np.zeros(npix) + np.nan
        unc_rhow = np.zeros((npix, nbands)) + np.nan

    #l2flags[np.array(oorFlagArray==1)] += 2**1 TODO flags?

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
        # Apply NNIOP to compute final IOPs
        log_iop2 = apply_backwardNN_TF(rho_w[:, iband_backwardNNIOP], sza, oza, nn_raa, valid, nnBackwardIOP_IO, NNIOP=True)
        iop_names = ['apig', 'adet', 'a_gelb', 'bpart', 'bwit']
        if scalar_dict is None:
            scalar_dict = {}
        for i, field in enumerate(iop_names):
            scalar_dict[field] = {'data': np.exp(log_iop2[:,i])}

    write_BalticP_AC_Product(product, baltic__product_path, sensor, spectral_dict, scalar_dict,
                             copyOriginalProduct, outputProductFormat, addName,
                             add_Idepix_Flags=add_Idepix_Flags, idepixProduct=idepixProduct,
                             add_L2Flags=add_L2Flags, L2FlagArray=l2flags,
                             add_Geometry=True, subset=subset)

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

def AC_forward(rho_rc, td, wavelength, sza, oza, nn_raa, valid, niop, Aatm, Aatm_inv):
    global n_evaluation
    n_evaluation = 0

    # Reference band
    wav_ref = float(bands_forwardNN[-1])
    iband_ref = iband_forwardNN[-1]

    # Define dimension
    n_dim = niop
    n_vertex = n_dim + 1
    nbands = len(bands_sat)
    n_pix  = np.count_nonzero(valid) # Apply NM only on valid pixel
    range_pix = np.arange(n_pix)
    range_keep = np.tile(range_pix,(n_vertex-1,1)).transpose()

    iter_max = 10
    n_iter_NM_max = 30
    n_iter_NM = 0
    simplex = np.ndarray((n_pix, n_vertex, n_dim))
    chi2_NM = np.ndarray((n_pix, n_vertex))
    rho_w_NM = np.ndarray((n_pix, n_vertex, nbands))
    rho_wmod_NM = np.ndarray((n_pix, n_vertex, nbands))
    rho_ag_NM = np.ndarray((n_pix, n_vertex, nbands))
    rho_ag_mod_NM = np.ndarray((n_pix, n_vertex, nbands))

    n_reflection = np.zeros(n_pix)
    n_expansion = np.zeros(n_pix)
    n_contraction =np.zeros(n_pix)
    n_reduction = np.zeros(n_pix)
    i_reduction = np.zeros(n_pix, dtype=bool)

    # Start NM - Define first vertice xbest of the simplex
    log_iop = apply_backwardNN_TF(rho_rc[:,iband_backwardNN], sza, oza, nn_raa, valid, nnBackward_IO)
    xbest = log_iop[valid,0:niop]

    while n_iter_NM < n_iter_NM_max:
        print("#### Iter NM %d"%n_iter_NM)

        # Update first simplex during overall NM iteration
        if n_iter_NM >=0:
            h = 0.02
            for iv in range(n_vertex):
                simplex[:, iv, :] = xbest[:, :]
                if iv >= 1:
                    simplex[:, iv, iv-1] *= 1. + h

        # Evaluate function at first simplex
        for iv in range(n_vertex):
            #simplex[:,iv] = check_and_constrain_iop(simplex[:,iv], nnForward_IO)
            chi2_NM[:,iv], rho_w_NM[:,iv], rho_wmod_NM[:,iv], rho_ag_NM[:,iv], rho_ag_mod_NM[:,iv] = evaluate_chi2(simplex[:,iv], rho_rc, td, wavelength, wav_ref, iband_ref, sza, oza, nn_raa, valid, Aatm_inv, Aatm)
                
        # Loop
        n_iter = 0
        n_reflection[:] = 0
        n_expansion[:] = 0
        n_contraction[:] = 0
        n_reduction[:] = 0
        while n_iter<iter_max:

            print("Iter %d"%n_iter)
            # Init action
            i_reduction[:] = False

            # Compute size of simplex (size of diam is n_pix)
            diam = diameter(simplex)
            if max(diam)< 1E-4:
                print("Diameter=%f -- break"%max(diam))
                break

            # Order vertices
            order = np.argsort(chi2_NM, axis=1)
            ibest = order[:,0] # best
            isecond = order[:,-2] # second to worse
            iworse = order[:,-1] # worse
            xbest = simplex[range_pix,ibest]
            xworse = simplex[range_pix,iworse]
            xnew = simplex[range_pix,ibest] # Values re-evaluated for expansion or contraction, but need realistic dummy for other
            ybest = chi2_NM[range_pix,ibest]
            ysecond = chi2_NM[range_pix,isecond]
            yworse = chi2_NM[range_pix,iworse]

            # Check convergence
            if max(ybest) < 1.E-6:
                print("Max(ybest)=%f -- break"%max(ybest))
                break
            
            # Compute centroid
            ikeep = order[:,:-1]
            xc = simplex[range_keep,ikeep].mean(1)

            # Reflect worse vertex
            xr = xc + (xc-xworse)
            # Evaluate reflection
            #xr = check_and_constrain_iop(xr, nnForward_IO)
            yr, rho_w_r, rho_wmod_r, rho_ag_r, rho_ag_mod_r = evaluate_chi2(xr, rho_rc, td, wavelength, wav_ref, iband_ref, sza, oza, nn_raa, valid, Aatm_inv, Aatm)

            # Replace vertex
            i_reflection = (ybest <= yr) & (yr <  ysecond)
            if i_reflection.any():
                simplex[range_pix[i_reflection],iworse[i_reflection]] = xr[i_reflection]
                chi2_NM[range_pix[i_reflection],iworse[i_reflection]] = yr[i_reflection]
                rho_w_NM[range_pix[i_reflection],iworse[i_reflection]] = rho_w_r[i_reflection]
                rho_wmod_NM[range_pix[i_reflection],iworse[i_reflection]] = rho_wmod_r[i_reflection]
                rho_ag_NM[range_pix[i_reflection],iworse[i_reflection]] = rho_ag_r[i_reflection]
                rho_ag_mod_NM[range_pix[i_reflection],iworse[i_reflection]] = rho_ag_mod_r[i_reflection]
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
                #xnew = check_and_constrain_iop(xnew, nnForward_IO)
                ynew, rho_w_new, rho_wmod_new, rho_ag_new, rho_ag_mod_new = evaluate_chi2(xnew, rho_rc, td, wavelength, wav_ref, iband_ref, sza, oza, nn_raa, valid, Aatm_inv, Aatm)

            # Apply expansion
            if i_expansion.any():
                i_e = i_expansion & (ynew <= yr)
                if i_e.any():
                    simplex[range_pix[i_e],iworse[i_e]] = xnew[i_e]
                    chi2_NM[range_pix[i_e],iworse[i_e]] = ynew[i_e]
                    rho_w_NM[range_pix[i_e],iworse[i_e]] = rho_w_new[i_e]
                    rho_wmod_NM[range_pix[i_e],iworse[i_e]] = rho_wmod_new[i_e]
                    rho_ag_NM[range_pix[i_e],iworse[i_e]] = rho_ag_new[i_e]
                    rho_ag_mod_NM[range_pix[i_e],iworse[i_e]] = rho_ag_mod_new[i_e]
                i_e = i_expansion & (ynew > yr)
                if i_e.any():
                    simplex[range_pix[i_e],iworse[i_e]] = xr[i_e]
                    chi2_NM[range_pix[i_e],iworse[i_e]] = yr[i_e]
                    rho_w_NM[range_pix[i_e],iworse[i_e]] = rho_w_r[i_e]
                    rho_wmod_NM[range_pix[i_e],iworse[i_e]] = rho_wmod_r[i_e]
                    rho_ag_NM[range_pix[i_e],iworse[i_e]] = rho_ag_r[i_e]
                    rho_ag_mod_NM[range_pix[i_e],iworse[i_e]] = rho_ag_mod_r[i_e]
                n_expansion[i_expansion] += 1 

            # Apply contraction
            if i_contraction.any():
                if i_ce.any():
                    ii = i_ce & (ynew <= yr)
                    if ii.any():
                        simplex[range_pix[ii],iworse[ii]] = xnew[ii]
                        chi2_NM[range_pix[ii],iworse[ii]] = ynew[ii]
                        rho_w_NM[range_pix[ii],iworse[ii]] = rho_w_new[ii]
                        rho_wmod_NM[range_pix[ii],iworse[ii]] = rho_wmod_new[ii]
                        rho_ag_NM[range_pix[ii],iworse[ii]] = rho_ag_new[ii]
                        rho_ag_mod_NM[range_pix[ii],iworse[ii]] = rho_ag_mod_new[ii]
                        n_contraction[ii] += 1
                    ii = i_ce & (ynew > yr)
                    if ii.any():
                        i_reduction[ii] = True
                if i_ci.any():
                    ii = i_ci & (ynew < yworse)
                    if ii.any():
                        simplex[range_pix[ii],iworse[ii]] = xnew[ii]
                        chi2_NM[range_pix[ii],iworse[ii]] = ynew[ii]
                        rho_w_NM[range_pix[ii],iworse[ii]] = rho_w_new[ii]
                        rho_wmod_NM[range_pix[ii],iworse[ii]] = rho_wmod_new[ii]
                        rho_ag_NM[range_pix[ii],iworse[ii]] = rho_ag_new[ii]
                        rho_ag_mod_NM[range_pix[ii],iworse[ii]] = rho_ag_mod_new[ii]
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
                    #simplex[i_reduction, iv] = check_and_constrain_iop(simplex[i_reduction, iv], nnForward_IO)
                    y, rho_w_tmp, rho_wmod_tmp, rho_ag_tmp, rho_ag_mod_tmp = evaluate_chi2(simplex[:,iv], rho_rc, td, wavelength, wav_ref, iband_ref, sza, oza, nn_raa, valid, Aatm_inv, Aatm)
                    chi2_NM[range_pix[i_reduction], iv] = y[i_reduction] # Care y defined only for dopix=i_reduction
                    rho_w_NM[range_pix[i_reduction], iv] = rho_w_tmp[i_reduction]
                    rho_wmod_NM[range_pix[i_reduction], iv] = rho_wmod_tmp[i_reduction]
                    rho_ag_NM[range_pix[i_reduction], iv] = rho_ag_tmp[i_reduction]
                    rho_ag_mod_NM[range_pix[i_reduction], iv] = rho_ag_mod_tmp[i_reduction]
                n_reduction[i_reduction] += 1


            # Release some memory
            del xworse, xnew, xc, xr
            del yr, rho_w_r, rho_wmod_r, rho_ag_r, rho_ag_mod_r
            if i_expansion.any() or i_contraction.any():
                del ynew, rho_w_new, rho_wmod_new, rho_ag_new, rho_ag_mod_new
            if i_reduction.any():
                del y, rho_w_tmp, rho_wmod_tmp, rho_ag_tmp, rho_ag_mod_tmp

            n_iter = n_iter + 1

        # End of algorithm
        print("Niter=",n_iter)
        print("Reflection=",n_reflection[0:10], np.count_nonzero(n_reflection))
        print("Expansion=",n_expansion[0:10], np.count_nonzero(n_expansion))
        print("Contraction=",n_contraction[0:10],  np.count_nonzero(n_contraction))
        print("Reduction=",n_reduction[0:10], np.count_nonzero(n_reduction))
        print("Evaluation=",n_evaluation)

        # Identify best vertex
        print("Identify best vertex")
        order = np.argsort(chi2_NM,axis=1)
        ibest = order[:,0] # best
        xbest = simplex[range_pix,ibest]
        #xbest = check_and_constrain_iop(xbest, nnForward_IO)

        n_iter_NM = n_iter_NM + 1

    # Final evaluation at best vertex
    print( "Evaluate best vertex")
    n_pix_all = oza.shape[0]
    chi2 = np.zeros(n_pix_all) + np.NaN
    rho_w = np.zeros((n_pix_all, nbands)) + np.NaN
    rho_wmod = np.zeros((n_pix_all, nbands)) + np.NaN
    rho_ag = np.zeros((n_pix_all, nbands)) + np.NaN
    rho_ag_mod = np.zeros((n_pix_all, nbands)) + np.NaN
    log_iop = np.zeros((n_pix_all, niop)) + np.NaN

    chi2[valid] = chi2_NM[range_pix,ibest]
    rho_w[valid] = rho_w_NM[range_pix,ibest]
    rho_wmod[valid] = rho_wmod_NM[range_pix,ibest]
    rho_ag[valid] = rho_ag_NM[range_pix,ibest]
    rho_ag_mod[valid] = rho_ag_mod_NM[range_pix,ibest]
    log_iop[valid] = xbest

    # Compute uncertainty on valid pixels, all bands
    print("Compute uncertainty")
    nband_chi2 = len(iband_chi2)
    # Compute Jacobian J = d rhow_mod/ d xw from simplex evaluation
    DX = np.zeros((n_pix, n_vertex - 1, n_dim))
    Drho_wmod = np.zeros((n_pix, n_vertex - 1, nbands))
    for v in range(1,n_vertex):
        DX[:,v-1] = simplex[range_pix, order[:,v]] - simplex[range_pix, order[:,0]]
        Drho_wmod[:,v-1] = rho_wmod_NM[range_pix, order[:,v]] - rho_wmod_NM[range_pix, order[:,0]]
    DXinv = np.linalg.inv(DX)
    J = np.zeros((n_pix, nband_chi2, niop))
    for k,ik in enumerate(iband_chi2):
        J[:,k] = np.einsum('...ij,...j->...i', DXinv, Drho_wmod[:,:,ik])
    # Identify non-singular pixels (J finite and inversible)
    non_singular_pix = np.all(np.isfinite(J), axis=(1,2))
    non_singular_pix[non_singular_pix] = np.linalg.matrix_rank(J[non_singular_pix]) == niop
    # Compute d rhorc_mod / d xw
    Matm = np.einsum('...ik,...kj->...ij', Aatm[valid], Aatm_inv[valid])
    TJ = np.einsum('...i,...ij->...ij', td[valid][:, iband_chi2], J)
    id_all = np.tile(np.identity(nband_chi2), (n_pix,1)).reshape(n_pix,nband_chi2,nband_chi2)
    drho_rcmod_dxw =  np.einsum('...ik,...kj->...ij', id_all - Matm[:,iband_chi2,:], TJ)
    # Compute variance-covariance matrix C_xw
    C_xw_inv = np.matmul(drho_rcmod_dxw.transpose(0,2,1),drho_rcmod_dxw)
    C_xw = np.zeros((n_pix, niop, niop)) + np.nan
    C_xw[non_singular_pix] = np.linalg.inv(C_xw_inv[non_singular_pix]) #TODO we could investigate inversion on non-singular dimensions < niop
    C_xw = np.einsum('i,ijk->ijk', chi2[valid], C_xw) # scaling with reduced chi2 for non-weighted minimization
    # Compute d rhow / d xw
    TinvMatm = np.einsum('...i,...ij->...ij', 1./td[valid], Matm)
    d_rhow_dxw = np.matmul(TinvMatm, TJ)
    # Compute variance-covariance matrix C_rhow
    C_rhow = np.matmul(d_rhow_dxw, np.matmul(C_xw, d_rhow_dxw.transpose(0,2,1)))
    # Return root square diagonal term
    unc_rhow = np.zeros((n_pix_all, nbands)) + np.NaN
    unc_rhow[valid] = np.sqrt(np.diagonal(C_rhow, axis1=1, axis2=2))

    l2flags = np.zeros(n_pix_all)
    # TODO define success = 
    #if not success:
    #    l2flags[ipix] += 2 ** 2
 
    return rho_w, rho_wmod, log_iop, rho_ag, rho_ag_mod, l2flags, chi2, unc_rhow

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
    all_valid = np.ones(n_pix, dtype=bool)

    # Compute rho_wmod
    # NN
    # Check iop range and apply constraints to forwardNN input range
    #log_iop = check_and_constrain_iop(vertices[index], nnForward_IO) # Don't apply it, gives worse results
    log_iop = vertices[index]
    rho_wmod = np.zeros((n_pix, nbands)) + np.NaN
    rho_wmod[:,iband_forwardNN] = apply_forwardNN_TF(log_iop, sza[valid][index], oza[valid][index], nn_raa[valid][index], all_valid)

    # Evaluate rho_ag
    rho_ag = rho_rc[valid][index] - td[valid][index]*rho_wmod

    # Fit rho_ag_mod
    x_atm = np.einsum('...ij,...j->...i', Aatm_inv[valid][index], rho_ag[:,iband_corr])
    x_atm_min = [-0.07,-0.01,-0.7]
    x_atm_max = [0.1,0.1,0.34]
    for i in range(3):
        x_atm[x_atm[:,i]<x_atm_min[i],i] = x_atm_min[i]
        x_atm[x_atm[:,i]>x_atm_max[i],i] = x_atm_max[i]

    # Compute rho_ag_mod
    rho_ag_mod = np.zeros((n_pix, nbands))# + np.NaN
    rho_ag_mod[:, :] = np.einsum('...ij,...j->...i', Aatm[valid][index], x_atm)

    # Compute rho_w
    rho_w = (rho_rc[valid][index] - rho_ag_mod)/td[valid][index]

    # Compute chi2
    chi2 = np.sum((td[valid][index][:, iband_chi2]*(rho_w[:,iband_chi2] - rho_wmod[:,iband_chi2]))**2, axis=1)
    #chi2 = np.sum((rho_w[:,iband_chi2] - rho_wmod[:,iband_chi2])**2, axis=1)
    #chi2 = np.sum(((rho_w[:,iband_chi2] - rho_wmod[:,iband_chi2])/rho_wmod[:,iband_chi2])**2, axis=1)
    # Normalise chi2 to number of degree of freedom (reduced chi-square)
    chi2 /= (len(iband_chi2) - vertices.shape[1])

    return chi2, rho_w, rho_wmod, rho_ag, rho_ag_mod

