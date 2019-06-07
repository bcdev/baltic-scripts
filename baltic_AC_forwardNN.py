#!/usr/bin/python
# -*- coding: utf-8 -*-
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
#sys.path.append('/home/cmazeran/.snap/snap-python')
sys.path.append("C:\\Users\Dagmar\Anaconda3\envs\py36\Lib\site-packages\snappy")
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
from baltic_corrections import gas_correction, glint_correction, white_caps_correction, Rayleigh_correction, diffuse_transmittance, Rmolgli_correction_Hygeos
import get_bands
from misc import default_ADF, nlinear
import luts_olci
import lut_hygeos
from auxdata_handling import setAuxData, checkAuxDataAvailablity

# Set locale for proper time reading with datetime
locale.setlocale(locale.LC_ALL, 'en_US.UTF_8')

# Select forward NN once for all
nnFilePath = "forwardNN_c2rcc/olci/olci_20171221/iop_rw/77x77x77_1798.8.net" # S3 OLCI
#nnFilePath = "forwardNN_c2rcc/msi/std_s2_20160502/iop_rw/17x97x47_125.5.net" # S2 MSI
NNffbpAlphaTabFast = jpy.get_type('org.esa.snap.core.nn.NNffbpAlphaTabFast')
nnfile = open(nnFilePath, 'r')
nnCode = nnfile.read()
nn_iop_rw = NNffbpAlphaTabFast(nnCode)

# Read NN input range once
def read_NN_input_ranges_fromFile():
    input_range = {}
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

inputRange = read_NN_input_ranges_fromFile()


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
      #  product.getTiePointGrid(name).readPixels(0, 0, width, height, var)
        #product.getTiePointGrid(name).getPixels(0, 0, width, height, var)
        var.shape = (height, width)
        for i in range(height):
            for j in range(width):
                var[i, j] = product.getTiePointGrid(name).getPixelDouble(i, j)
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

def read_NN_metadata(nnpath):
    ##
    # read the metadata:
    # nnpath = 'D:\WORK\IdePix\\NN_training_S2\I13x11x9x6x4x3xO1_sqrt_Radical2TrainingSelection_Relu_NoScaler\\'
    meta_fnames = os.listdir(nnpath)
    meta_fn = [fn for fn in meta_fnames if 'Metadata_' in fn]
    with open(nnpath + meta_fn[0], "r") as f:
        d = f.read()
    training_meta = json.loads(d)
    f.close()
    model_fn = [fn for fn in meta_fnames if 'MetadataModel_' in fn]
    with open(nnpath + model_fn[0], "r") as f:
        d = f.read()
    model_meta = json.loads(d)
    f.close()
    return training_meta, model_meta

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
    elif sensor == 'S2MSI': #TODO, after S2_resampling.
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


def check_valid_pixel_expression_L1(product, sensor):
    if sensor == 'OLCI':
        height = product.getSceneRasterHeight()
        width = product.getSceneRasterWidth()
        quality_flags = np.zeros(width * height, dtype='uint32')
        product.getBand('quality_flags').readPixels(0, 0, width, height, quality_flags)

        # Masks OLCI L1
        ## flags: 31=land 30=coastline 29=fresh_inland_water 28=tidal_region 27=bright 26=straylight_risk 25=invalid
        ## 24=cosmetic 23=duplicated 22=sun-glint_risk 21=dubious 20->00=saturated@Oa01->saturated@Oa21
        invalid_mask = np.bitwise_and(quality_flags, 2 ** 25) == 2 ** 25
        land_mask = (np.bitwise_and(quality_flags, 2 ** 31) == 2 ** 31) | \
                                (np.bitwise_and(quality_flags, 2 ** 30) == 2 ** 30)
                                #(np.bitwise_and(quality_flags, 2 ** 29) == 2 ** 29)
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

# def apply_forwardNN_IOP_to_rhow_keras(X, sensor):
#     start_time = time.time()
#     ###
#     # read keras NN + metadata
#     NN_path = '...'  # full path to NN file.
#     metaNN_path = '...'  # folder with metadata files from training
#     model = load_model(NN_path)
#     training_meta, model_meta = read_NN_metadata(metaNN_path)
#
#     X_trans = np.copy(X)
#     ###
#     # transformation of input data
#     transform_method = training_meta['transformation_method']
#     if transform_method == 'sqrt':
#         X_trans = np.sqrt(X_trans)
#     elif transform_method == 'log':
#         X_trans = np.log10(X_trans)
#
#     ###
#     if model_meta['scaling']:
#         scaler_path = os.listdir(metaNN_path)
#         scaler_path = [sp for sp in scaler_path if 'scaling' in sp][0]
#         print(scaler_path)
#         scaler = pd.read_csv(metaNN_path + '/' + scaler_path, header=0, sep="\t", index_col=0)
#         for i in range(X.shape[1]):
#             X_trans[:, i] = (X_trans[:, i] - scaler['mean'].loc[i]) / scaler['var'].loc[i]
#
#     ###
#     # Application of the NN to the data.
#     prediction = model.predict(X_trans)
#     print(len(prediction.shape))
#
#     print("model load, transform, predict: %s seconds " % round(time.time() - start_time, 2))
#     return prediction

def apply_forwardNN_IOP_to_rhow(iop, sun_zenith, view_zenith, diff_azimuth, sensor, valid_data, T=15, S=35, nn=''):
    """
    Apply the forwardNN: IOP to rhow
    input: numpy array iop, shape = (Npixels x iops= (log_apig, log_adet, log a_gelb, log_bpart, log_bwit)),
            np.array sza, shape = (Npixels,)
            np.array oza, shape = (Npixels,)
            np.array raa, shape = (Npixels,); range: 0-180
    returns: np.array rhow, shape = (Npixels, wavelengths)

    T, S: currently constant #TODO take ECMWF temperature at sea surface?
    valid ranges can be found at the beginning of the .net-file.

    NN output (12 bands): log_rw at lambda = 400, 412, 443, 489, 510, 560, 620, 665, 674, 681, 709, 754
    """

    # Initialise output
    nBands = None
    if sensor == 'OLCI':
        nBands = 12
    elif sensor == 'S2':
        nBands = 6
    output = np.zeros((iop.shape[0], nBands)) + np.NaN

    ###
    # Launch the NN
    # Important:
    # OLCI input array has to be of size 10: [SZA, VZA, RAA, T, S, log_apig, log_adet, log a_gelb, log_bpart, log_bwit]
    # S2 input array has to be of size 10 (same order as OLCI): [ sun_zeni, view_zeni, azi_diff, T, S, log_conc_apig, log_conc_adet,
    # log_conc_agelb, log_conc_bpart, log_conc_bwit]
    inputNN = np.zeros(10)
    inputNN[3] = T
    inputNN[4] = S
    for i in range(iop.shape[0]):
        if valid_data[i]:
            inputNN[0] = sun_zenith[i]
            inputNN[1] = view_zenith[i]
            inputNN[2] = diff_azimuth[i]
            for j in range(iop.shape[1]): # log_apig, log_adet, log a_gelb, log_bpart, log_bwit
                inputNN[j+5] = iop[i, j]
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

def write_BalticP_AC_Product_old(product, baltic__product_path, sensor, spectral_dict, scalar_dict=None):
    # Initialise the output product
    File = jpy.get_type('java.io.File')
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    bandShape = (height, width)

    balticPACProduct = Product('balticPAC', 'balticPAC', width, height)
    balticPACProduct.setFileLocation(File(baltic__product_path))

    ProductUtils.copyGeoCoding(product, balticPACProduct)
    ProductUtils.copyTiePointGrids(product, balticPACProduct)

    # Define total number of bands (TOA)
    if (sensor == 'OLCI'):
        nbands = 21
    #elif (sensor == 'S2'): TODO

    # Get TOA radiance sources for spectral band properties
    bsources = [product.getBand("Oa%02d_radiance"%(i+1)) for i in range(nbands)]

    # Create empty bands for spectral fields (note: number of spectral bands may differ)
    for key in spectral_dict.keys():
        data = spectral_dict[key].get('data')
        if not data is None:
            nbands_key = data.shape[-1]
            for i in range(nbands_key):
                brtoa_name = key + "_" + str(i + 1)
                rtoaBand = balticPACProduct.addBand(brtoa_name, ProductData.TYPE_FLOAT32)
                ProductUtils.copySpectralBandProperties(bsources[i], rtoaBand)
                rtoaBand.setNoDataValue(np.nan)
                rtoaBand.setNoDataValueUsed(True)

    # Set auto grouping
    autoGroupingString = ':'.join(spectral_dict.keys())
    balticPACProduct.setAutoGrouping(autoGroupingString)

    # Create empty bands for scalar fields
    if not scalar_dict is None:
        for key in scalar_dict.keys():
            singleBand = balticPACProduct.addBand(key, ProductData.TYPE_FLOAT32)
            singleBand.setNoDataValue(np.nan)
            singleBand.setNoDataValueUsed(True)

    # Initialise writer
    writer = ProductIO.getProductWriter('BEAM-DIMAP')
    #writer = ProductIO.getProductWriter('CSV')
    balticPACProduct.setProductWriter(writer)
    balticPACProduct.writeHeader(baltic__product_path)
    writer.writeProductNodes(balticPACProduct, baltic__product_path)

    #Write Latitude, Longitude explicitly
    geoBand = balticPACProduct.getBand('longitude')
    geoBand.writeRasterDataFully()
    geoBand = balticPACProduct.getBand('latitude')
    geoBand.writeRasterDataFully()

    # Write data of spectral fields
    for key in spectral_dict.keys():
        data = spectral_dict[key].get('data')
        if not data is None:
            nbands_key = data.shape[-1]
            for i in range(nbands_key):
                brtoa_name = key + "_" + str(i + 1)
                rtoaBand = balticPACProduct.getBand(brtoa_name)
                out = np.array(data[:, i]).reshape(bandShape)
                rtoaBand.writeRasterData(0, 0, width, height, snp.ProductData.createInstance(np.float32(out)),
                                                                         ProgressMonitor.NULL)

    # Write data of scalar fields
    if not scalar_dict is None:
        for key in scalar_dict.keys():
            data = scalar_dict[key].get('data')
            if not data is None:
                singleBand = balticPACProduct.getBand(key)
                out = np.array(data).reshape(bandShape)
                singleBand.writeRasterData(0, 0, width, height, snp.ProductData.createInstance(np.float32(out)),
                                                                          ProgressMonitor.NULL)


    # # Create flag coding
    # raycorFlagsBand = balticPACProduct.addBand('raycor_flags', ProductData.TYPE_UINT8)
    # raycorFlagCoding = FlagCoding('raycor_flags')
    # raycorFlagCoding.addFlag("testflag_1", 1, "Flag 1 for Rayleigh Correction")
    # raycorFlagCoding.addFlag("testflag_2", 2, "Flag 2 for Rayleigh Correction")
    # group = balticPACProduct.getFlagCodingGroup()
    # group.add(raycorFlagCoding)
    # raycorFlagsBand.setSampleCoding(raycorFlagCoding)

    balticPACProduct.closeIO()


def write_BalticP_AC_Product(product, baltic__product_path, sensor, spectral_dict, scalar_dict=None, copyOriginalProduct=False, outputProductFormat="BEAM-DIMAP"):
    # Initialise the output product
    File = jpy.get_type('java.io.File')
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    bandShape = (height, width)

    if outputProductFormat == "BEAM-DIMAP":
        baltic__product_path = baltic__product_path + '.dim'
    elif outputProductFormat == 'CSV':
        baltic__product_path = baltic__product_path + '.csv'

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
            print(nbands_key)
            if outputProductFormat == 'BEAM-DIMAP':
                if sensor == 'OLCI':
                    bsources = [product.getBand("Oa%02d_radiance" % (i + 1)) for i in range(nbands)]
                elif sensor == 'S2MSI':
                    bsources = [product.getBand("B%d" % (i + 1)) for i in range(8)]
                    bsources.append(product.getBand('B8A'))
                    [bsources.append(product.getBand("B%d" % (i + 9))) for i in range(4)]

            for i in range(nbands_key):
                brtoa_name = key + "_" + str(i + 1)
                print(brtoa_name)
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
            print(key)
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
    taum = adf_ppp.tau_ray #TODO use per-pixel wavelength
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

def check_and_constrain_iop(iop):
    for i, varn in enumerate(['log_apig', 'log_adet', 'log_agelb', 'log_bpart', 'log_bwit']):
        mi = inputRange[varn][0]
        ma = inputRange[varn][1]
        if iop[i] < mi:
            iop[i] = mi
        if iop[i] > ma:
            iop[i] = ma
    return iop


def ac_cost(iop, sensor, nbands, iband_NN, iband_corr, iband_chi2, rho_rc, td, sza, oza, raa, Aatm, Aatm_inv, valid):
    """
    Cost function to be minimized, define for one pixel
    """

    # Compute rhow_mod
    rho_wmod = np.zeros(nbands) + np.NaN

    # Check iop range and apply constraints to forwardNN input range
    iop = check_and_constrain_iop(iop)

    rho_wmod[iband_NN] = apply_forwardNN_IOP_to_rhow(np.array([iop]), np.array([sza]), np.array([oza]), np.array([raa]), sensor,np.array([valid]))
    # Compute rho_ag and fit best atmospheric model
    rho_ag = rho_rc - td*rho_wmod
    coefs = np.einsum('...ij,...j->...i', Aatm_inv, rho_ag[iband_corr])
    rho_ag_mod  = np.einsum('...ij,...j->...i',Aatm,coefs)
    # Compute rho_w
    rho_w = (rho_rc - rho_ag_mod)/td
    # Compute residual and chi2
    res  = rho_w[iband_chi2]-rho_wmod[iband_chi2] # TODO one other option is to minimize at TOA by multiplying by td
    chi2 = np.sum(res*res) # TODO cost function should include weighting; option in relative difference
    return chi2


def baltic_AC_forwardNN(scene_path='', filename='', outpath='', sensor='', subset=None, addName = '', outputSpectral=None,
                        outputScalar=None, correction='HYGEOS', copyOriginalProduct=False, outputProductFormat="BEAM-DIMAP",
                        atmosphericAuxDataPath = None):
    """
    Main function to run the Baltic+ AC based on forward NN
    correction: 'HYGEOS' or 'IPF' for Rayleigh+glint correction
    """

    # Option for IPF or HYGEOS glint+Rayleigh correction: 'IPF' or 'HYGEOS'
    #correction = 'HYGEOS'
    #correction = 'IPF'

    # Get sensor & AC bands
    bands_sat, bands_rw, bands_corr, bands_chi2, bands_forwardNN, bands_abs = get_bands.main(sensor,"dummy")
    nbands = len(bands_sat)
    iband_corr = np.searchsorted(bands_sat, bands_corr)
    iband_chi2 = np.searchsorted(bands_sat, bands_chi2)
    iband_NN = np.searchsorted(bands_sat, bands_forwardNN)
    iband_abs = np.searchsorted(bands_sat, bands_abs)

    # Initialising a product for Reading with snappy

    product = snp.ProductIO.readProduct(os.path.join(scene_path, filename))

    if sensor == "S2MSI" : # todo: and if product is not resampled!
        # Resampling to 60m
        parameters = HashMap()
        parameters.put('resolution', '60')
        parameters.put('upsampling', 'Bicubic')
        parameters.put('downsampling', 'Mean') #
        parameters.put('flagDownsampling', 'FlagOr') # 'First', 'FlagAnd'
        product = GPF.createProduct('S2Resampling', parameters, product)


    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    npix = width*height

    # Read L1B product and convert Radiance to reflectance
    rho_toa = radianceToReflectance_Reader(product, sensor=sensor)


    # Classify pixels with Level-1 flags
    valid = check_valid_pixel_expression_L1(product, sensor)

    # # Limit processing to sub-box
    # if subset:
    #     sline,eline,scol,ecol = subset
    #     valid_subset = np.zeros((height,width),dtype='bool')
    #     valid_subset[sline:eline+1,scol:ecol+1] = valid.reshape(height,width)[sline:eline+1,scol:ecol+1]
    #     print('subset, valid:', np.nansum(valid_subset))
    #     valid = valid_subset.reshape(height*width)
    #     del valid_subset
    #
    # Read geometry and compute relative azimuth angle
    oaa, oza, saa, sza = angle_Reader(product, sensor)
    raa, nn_raa = calculate_diff_azim(oaa, saa)

    if sensor == 'OLCI':
        # Read per-pixel wavelength
        wavelength = Level1_Reader(product, sensor, 'lambda0', reshape=False)
        # Read latitude, longitude
        latitude = get_band_or_tiePointGrid(product, 'latitude', reshape=False)
        longitude = get_band_or_tiePointGrid(product, 'longitude', reshape=False)
    elif sensor == 'S2MSI': #todo
        wavelength = []
        # Read latitude, longitude
       # latitude = get_band_or_tiePointGrid(product, 'Latitude', reshape=False)
       # longitude = get_band_or_tiePointGrid(product, 'Longitude', reshape=False)

    # Read day in the year
    yday = get_yday(product, reshape=False)
    print(yday)

    if sensor == 'OLCI':
        # Read meteo data
        pressure = get_band_or_tiePointGrid(product, 'sea_level_pressure', reshape=False)
        ozone = get_band_or_tiePointGrid(product, 'total_ozone', reshape=False)
        tcwv = get_band_or_tiePointGrid(product, 'total_columnar_water_vapour', reshape=False)
        wind_u = get_band_or_tiePointGrid(product, 'horizontal_wind_vector_1', reshape=False)
        wind_v = get_band_or_tiePointGrid(product, 'horizontal_wind_vector_2', reshape=False)
        windm = np.sqrt(wind_u*wind_u+wind_v*wind_v)
    elif sensor == 'S2MSI':
        default_ozone = 330. #DU
        default_pressure = 1000. #hPa

        if atmosphericAuxDataPath != None:
            checkAuxDataAvailablity(atmosphericAuxDataPath, product)
            ozone, pressure, tcwv, wind_u, wind_v = setAuxData(product, atmosphericAuxDataPath)



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

    # Compute diffuse transmittance (Rayleigh)
    td = diffuse_transmittance(sza, oza, pressure, adf_ppp)

    # Glint correction
    rho_g, rho_gc = glint_correction(rho_ng, valid, sza, oza, saa, raa, pressure, wind_u, wind_v, windm, adf_ppp)

    if correction == 'IPF':
        # Glint correction
        rho_g, rho_gc = glint_correction(rho_ng, valid, sza, oza, saa, raa, pressure, wind_u, wind_v, windm, adf_ppp)

        # White-caps correction
        #rho_wc, rho_gc = white_caps_correction(rho_ng, valid, windm, td, adf_ppp)

        # Rayleigh correction
        rho_r, rho_rc = Rayleigh_correction(rho_gc, valid, sza, oza, raa, pressure, windm, adf_acp, adf_ppp, sensor)

    elif correction == 'HYGEOS':
        # Glint correction - required only to get rho_g in the AC
        rho_g, dummy = glint_correction(rho_ng, valid, sza, oza, saa, raa, pressure, wind_u, wind_v, windm, adf_ppp)

        #LUT_HYGEOS = lut_hygeos.LUT('./auxdata/LUT.hdf')
        # Glint + Rayleigh correction
        rho_r, rho_molgli, rho_rc = Rmolgli_correction_Hygeos(rho_ng, valid, latitude, sza, oza, raa, wavelength, pressure, windm, LUT_HYGEOS)
        #rho_r, rho_molgli, rho_rc = Rmolgli_correction_Hygeos(rho_ng, valid, latitude, sza, oza, raa, wavelength,
        #													  pressure, windm, lut_hygeos)


    # Atmospheric model
    print("Compute atmospheric matrices")
    Aatm, Aatm_inv = polymer_matrix(bands_sat,bands_corr,valid,rho_g,rho_r,sza,oza,wavelength,adf_ppp)

    # Inversion of iop = [log_apig, log_adet, log a_gelb, log_bpart, log_bwit]
    print("Inversion")
    niop = 5
    iop = np.zeros((npix,niop)) + np.NaN
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
        #iop_0 = np.array([-4.3414865, -4.956355, - 3.7658699, - 1.8608053, - 2.694404])
        iop_0 = np.array([-3., -3., -3., -3., -3.])
        # Nelder-Mead optimization
        #args_pix = (sensor, nbands, iband_NN, iband_corr, iband_chi2, rho_rc[ipix], td[ipix], sza[ipix], oza[ipix], raa[ipix], Aatm[ipix], Aatm_inv[ipix], valid[ipix])
        args_pix = (sensor, nbands, iband_NN, iband_corr, iband_chi2, rho_rc[ipix], td[ipix], sza[ipix], oza[ipix], nn_raa[ipix], Aatm[ipix], Aatm_inv[ipix], valid[ipix])
        NM_res = minimize(ac_cost,iop_0,args=args_pix,method='nelder-mead')#, options={'maxiter':150', disp': True})
        iop[ipix,:] = NM_res.x
        #success = NM_res.success TODO add a flag in the Level-2 output to get this info
        ipix_proc += 1
    print("")

    # Compute the final residual
    print("Compute final residual")
    rho_wmod = np.zeros((npix, nbands)) + np.NaN
    #rho_wmod[:,iband_NN] = apply_forwardNN_IOP_to_rhow(iop, sza, oza, raa, sensor,valid)
    rho_wmod[:, iband_NN] = apply_forwardNN_IOP_to_rhow(iop, sza, oza, nn_raa, sensor, valid)
    rho_ag_mod = np.zeros((npix, nbands)) + np.NaN
    rho_ag = rho_rc - td*rho_wmod
    coefs = np.einsum('...ij,...j->...i', Aatm_inv[valid], rho_ag[valid][:,iband_corr])
    rho_ag_mod[valid] = np.einsum('...ij,...j->...i',Aatm[valid],coefs)
    rho_w = (rho_rc - rho_ag_mod)/td

    # Set absorption band to NaN
    rho_w[:,iband_abs] = np.NaN

    # TODO Normalisation
    # rho_wn =

    # TODO uncertainties
    # unc_rhow =

    ###
    # Writing a product
    # input: spectral_dict holds the spectral fields
    #        scalar_dict holds the scalar fields
    ###
    baltic__product_path = os.path.join(outpath,'baltic_' + filename + addName)
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

    write_BalticP_AC_Product(product, baltic__product_path, sensor, spectral_dict, scalar_dict, copyOriginalProduct, outputProductFormat)

    product.closeProductReader()



