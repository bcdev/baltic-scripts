import os
import glob
import numpy as np
import sys

sys.path.append("C:\\Users\Dagmar\Anaconda3\envs\py36\Lib\site-packages\snappy")
import snappy as snp
from snappy import jpy
from snappy import PixelPos
from snappy import ProductDataUTC
#in order to run this, you have to add the following line to your __init__.py in your snappy:
# ProductDataUTC = jpy.get_type('org.esa.snap.core.datamodel.ProductData$UTC')


Calendar = jpy.get_type('java.util.Calendar')
AncDownloader = jpy.get_type('org.esa.s3tbx.c2rcc.ancillary.AncDownloader')
File = jpy.get_type('java.io.File')

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


def getGeoPositionsForS2Product(product):
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()

    latitude = np.zeros((height, width))
    longitude = np.zeros((height, width))

    for x in range(width):
        for y in range(height):
            pixelPos = PixelPos(x + 0.5, y + 0.5)
            geoPos = product.getSceneGeoCoding().getGeoPos(pixelPos, None)
            latitude[y,x] = geoPos.getLat()
            longitude[y, x] = geoPos.getLon()

    return latitude, longitude


def findProductCornerInAuxData(product):
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()

    cornerX = [0 + 0.5, 0 + 0.5, width - 0.5, width - 0.5]
    cornerY = [0 + 0.5, height - 0.5, 0 + 0.5, height - 0.5]
    cornerLat = []
    cornerLon = []
    for x, y in zip(cornerX, cornerY):
        pixelPos = PixelPos(x + 0.5, y + 0.5)
        geoPos = product.getSceneGeoCoding().getGeoPos(pixelPos, None)
        cornerLat.append(geoPos.getLat() + 90.)  # only positive indeces!
        if geoPos.getLon() < 0:
            cornerLon.append(360 - geoPos.getLon())
        else:
            cornerLon.append(geoPos.getLon())

    minLat = np.min(cornerLat)
    maxLat = np.max(cornerLat)
    minLon = np.min(cornerLon)
    maxLon = np.max(cornerLon)

    coordXmin = np.floor(minLon)
    coordXmax = np.round(maxLon)
    if coordXmax < maxLon:
        coordXmax += 1

    coordYmin = np.floor(minLat)
    coordYmax = np.round(maxLat)
    if coordYmax < maxLat:
        coordYmax += 1

    return coordXmin.astype(np.int32), coordXmax.astype(np.int32), coordYmin.astype(np.int32), coordYmax.astype(np.int32)


def setAuxData(product, AuxFullFilePath_dict):

    # #
    # m_wind, z_wind, p_water -> precipitable water [kg/m^2], rel_hum, press [mbar]
    # met_fname_parts = ["_MET_NCEPR2_6h.hdf",
    #                    "_MET_NCEPN_6h.hdf"]  # "_MET_NCEPR2_6h.hdf.bz2", "_MET_NCEPN_6h.hdf.bz2"
    # metFiles = []
    # for fn in files:
    #     for pf in met_fname_parts:
    #         if pf in fn:
    #             metFiles.append(fn)
    #
    # ozone_fname_parts = ["_O3_TOMSOMI_24h.hdf", "_O3_N7TOMS_24h.hdf", "_O3_EPTOMS_24h.hdf", "_O3_AURAOMI_24h.hdf"]
    # # "_O3_TOMSOMI_24h.hdf.bz2", "_O3_N7TOMS_24h.hdf.bz2", "_O3_EPTOMS_24h.hdf.bz2","_O3_AURAOMI_24h.hdf.bz2"
    # ozoneFiles = []
    # for fn in files:
    #     for pf in ozone_fname_parts:
    #         if pf in fn:
    #             ozoneFiles.append(fn)
    #
    # Extraction and Interpolation of ozone and pressure values, by lat, lon; by time.
    # Extract: corners of the product; everything inbetween and surrounding the outlines;
    # If the values are not changing that much, set to fixed value.
    # Else: Interpolate for all pixels on this small dataset.

    auxData = {
        'ozone': 330. , # DU
        'pressure': 1000., #hPa
        'tcwv': 0.,
        'wind_u': 0.,
        'wind_v': 0.
    }

    print(AuxFullFilePath_dict['ozone'][0])
    ozoneStartProduct = snp.ProductIO.readProduct(AuxFullFilePath_dict['ozone'][0])
    ozoneEndProduct = snp.ProductIO.readProduct(AuxFullFilePath_dict['ozone'][1])

    ozoneStart = get_band_or_tiePointGrid(ozoneStartProduct, 'ozone', reshape=True)
    ozoneEnd = get_band_or_tiePointGrid(ozoneEndProduct, 'ozone', reshape=True)
    ozoneStartProduct.closeProductReader()
    ozoneEndProduct.closeProductReader()

    coordXmin, coordXmax, coordYmin, coordYmax = findProductCornerInAuxData(product)

    a = ozoneStart[coordYmin:coordYmax, coordXmin:coordXmax]
    print(a)
    b = ozoneEnd[coordYmin:coordYmax, coordXmin:coordXmax]
    print(b)
    ##
    # averaging the ozone values spatially and temporally...
    auxData['ozone'] = np.mean((a+b)/2.)

    METStartProduct = snp.ProductIO.readProduct(AuxFullFilePath_dict['MET'][0])
    METEndProduct = snp.ProductIO.readProduct(AuxFullFilePath_dict['MET'][1])

    met_variable_names = {'pressure':'press', 'wind_u': 'z_wind', 'wind_v': 'm_wind', 'tcwv':'p_water'}

    for key in met_variable_names.keys():
        metStart = get_band_or_tiePointGrid(METStartProduct, met_variable_names[key], reshape=True)
        metEnd = get_band_or_tiePointGrid(METEndProduct,  met_variable_names[key], reshape=True)
        a = metStart[coordYmin:coordYmax, coordXmin:coordXmax]
        b = metEnd[coordYmin:coordYmax, coordXmin:coordXmax]
        ##
        # averaging the MET values spatially and temporally...
        auxData[key] = np.mean((a + b)/2.)

    METStartProduct.closeProductReader()
    METEndProduct.closeProductReader()

    return auxData


def yearAndDoyAndHourUTC(dateMJD):
    utc = ProductDataUTC(dateMJD)
    calendar = utc.getAsCalendar()
    year = calendar.get(Calendar.YEAR)
    doy = calendar.get(Calendar.DAY_OF_YEAR)
    h = calendar.get(Calendar.HOUR_OF_DAY)
    return year, doy, h

def convertToFileNamePrefix(dateMJD):
    year, doy, h = yearAndDoyAndHourUTC(dateMJD)
    prefix = "N%4d%03d%02d" % (year, doy, h)
    return prefix

def StartAndEndFileTimeMJD24H(timeMJD): # for daily ozone
    startFileTimeMJD = np.floor(timeMJD - 0.5)
    return startFileTimeMJD, startFileTimeMJD +1

def InterpolationBorderComputer24H(timeMJD): # for daily ozone
    startFileTimeMJD = np.floor(timeMJD - 0.5)
    StartBorderTimeMDJ = startFileTimeMJD + 0.5
    StartAncFilePrefix = convertToFileNamePrefix(startFileTimeMJD)
    EndAncFilePrefix = convertToFileNamePrefix(startFileTimeMJD + 1)
    return StartAncFilePrefix, EndAncFilePrefix

def StartAndEndFileTimeMJD6H(timeMJD):
    startFileTimeMJD = np.floor((timeMJD - 0.125) * 4) * 0.25
    return startFileTimeMJD, startFileTimeMJD + 0.25

def InterpolationBorderComputer6H(timeMJD):
    startFileTimeMJD = np.floor((timeMJD - 0.125) * 4) * 0.25
    StartBorderTimeMDJ = startFileTimeMJD + 0.125
    EndBorderTimeMJD = StartBorderTimeMDJ + 0.25
    StartAncFilePrefix = convertToFileNamePrefix(startFileTimeMJD)
    EndAncFilePrefix = convertToFileNamePrefix(startFileTimeMJD + 0.25)
    return StartAncFilePrefix, EndAncFilePrefix

def checkAuxDataAvailablity(pathAuxDataRep, product):
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()

    startTime = product.getStartTime()
    # dateString = startTime.getElemString()
    # date = startTime.getAsDate().getTime()
    # year = startTime.getAsDate().getYear() + 1900

    download_path = "http://oceandata.sci.gsfc.nasa.gov/cgi/getfile/"

    dateMJD = startTime.getMJD()
    ###
    # for ozone:
    startTime, endTime = StartAndEndFileTimeMJD24H(dateMJD)
    startFilePrefix, endFilePrefix = InterpolationBorderComputer24H(dateMJD)

    # does ozone file exist?
    if os.path.exists(pathAuxDataRep):
        files = glob.glob(pathAuxDataRep + '/**/*.hdf*', recursive=True)

        year, doy, h = yearAndDoyAndHourUTC(startTime)
        startFilePath = pathAuxDataRep + str(year) + '\\'+ "%03d" % doy + '\\'

        if os.path.exists(startFilePath):
            fullStartFilePathOzone = [fn for fn in files if startFilePath + startFilePrefix in fn and 'O3' in fn]
        else:
            # download the missing ozone data!!
            print('Warning: Ozone data for first day is missing. Automatic download not yet implemented.')
            fullStartFilePathOzone = ''

        year, doy, h = yearAndDoyAndHourUTC(endTime)
        startFilePath = pathAuxDataRep + str(year) + '\\' + "%03d" % doy + '\\'
        if os.path.exists(startFilePath):
            fullEndFilePathOzone = [fn for fn in files if startFilePath + endFilePrefix in fn and 'O3' in fn]
        else:
            # download the missing ozone data!!
            print('Warning: Ozone data for second day is missing. Automatic download not yet implemented.')
            fullEndFilePathOzone = ''

    ###
    # for meteorology:
    startTime, endTime = StartAndEndFileTimeMJD6H(dateMJD)
    startFilePrefix, endFilePrefix = InterpolationBorderComputer6H(dateMJD)

    # does meteorology file exist?
    if os.path.exists(pathAuxDataRep):
        files = glob.glob(pathAuxDataRep + '/**/*.hdf*', recursive=True)

        year, doy, h = yearAndDoyAndHourUTC(startTime)
        startFilePath = pathAuxDataRep + str(year) + '\\' + "%03d" % doy + '\\'
        if os.path.exists(startFilePath):
            fullStartFilePathMET = [fn for fn in files if startFilePath + startFilePrefix in fn and not '.bz2' in fn]
        else:
            # download the missing meteorology data!!
            fullStartFilePathMET = ''

        year, doy, h = yearAndDoyAndHourUTC(endTime)
        startFilePath = pathAuxDataRep + str(year) + '\\' + "%03d" % doy + '\\'
        if os.path.exists(startFilePath):
            fullEndFilePathMET = [fn for fn in files if startFilePath + endFilePrefix in fn and not '.bz2' in fn]
        else:
            # download the missing meteorology data!!
            fullEndFilePathMET = ''


    #ancDownloader = AncDownloader(download_path)
    #ancDownloader.download(File(fullEndFilePathMET))
    # ancRepository = AncRepository( File(atmosphericAuxDataPath), ancDownloader)
    # ozone = np.array(330., dtype='float64')
    # surfacePressure = np.array(1000., dtype='float64')
    # ozoneFormat = AncillaryCommons.createOzoneFormat(ozone)
    # pressureFormat = AncillaryCommons.createPressureFormat(surfacePressure)
    # auxdata = new AtmosphericAuxdataDynamic(ancRepository, ozoneFormat, pressureFormat)
    out_dict = {
        'ozone': [fullStartFilePathOzone[0], fullEndFilePathOzone[0]],
        'MET': [fullStartFilePathMET[0], fullEndFilePathMET[0]]
    }

    return out_dict