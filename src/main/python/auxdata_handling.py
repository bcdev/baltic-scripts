import os
import glob
import numpy as np
import sys
#import requests
import datetime
from datetime import date

#sys.path.append('/home/cmazeran/.snap/snap-python')
# sys.path.append("C:\\Users\Dagmar\Anaconda3\envs\py36\Lib\site-packages\snappy")
sys.path.append("C:\\Users\Dagmar\.snap\snap-python")
import snappy as snp
from snappy import jpy
from snappy import PixelPos
#from snappy import ProductDataUTC
#in order to run this, you have to add the following line to your __init__.py in your snappy:
# ProductDataUTC = jpy.get_type('org.esa.snap.core.datamodel.ProductData$UTC')


Calendar = jpy.get_type('java.util.Calendar')
AncDownloader = jpy.get_type('org.esa.s3tbx.c2rcc.ancillary.AncDownloader')
File = jpy.get_type('java.io.File')

def recursive_glob(rootdir='.', suffixes=''):
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if True in [filename.endswith(suffix) for suffix in suffixes]]

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

def findSinglePositionInAuxData(Lat, Lon):

    cornerLat = []
    cornerLon = []
    cornerLat.append(Lat + 90.)  # only positive indeces!
    if Lon < 0:
        cornerLon.append(360 - Lon)
    else:
        cornerLon.append(Lon)

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

def setAuxData(product, AuxFullFilePath_dict, singlePosition=False, Lat=None, Lon=None):

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
        'tcwv': 10.,
        'wind_u': 0.,
        'wind_v': 0.
    }

    OzoneStartFileValid = os.path.getsize(AuxFullFilePath_dict['ozone'][0]) > 1000
    OzoneEndFileValid = os.path.getsize(AuxFullFilePath_dict['ozone'][1]) > 1000
    if OzoneStartFileValid:
        ozoneStartProduct = snp.ProductIO.readProduct(AuxFullFilePath_dict['ozone'][0])
        ozoneStart = get_band_or_tiePointGrid(ozoneStartProduct, 'ozone', reshape=True)
        ozoneStartProduct.closeProductReader()
    if OzoneEndFileValid:
        ozoneEndProduct = snp.ProductIO.readProduct(AuxFullFilePath_dict['ozone'][1])
        ozoneEnd = get_band_or_tiePointGrid(ozoneEndProduct, 'ozone', reshape=True)
        ozoneEndProduct.closeProductReader()

    if singlePosition:
        coordXmin, coordXmax, coordYmin, coordYmax = findSinglePositionInAuxData(Lat, Lon)
    else:
        coordXmin, coordXmax, coordYmin, coordYmax = findProductCornerInAuxData(product)

    a = None
    b = None
    if OzoneStartFileValid:
        a = ozoneStart[coordYmin:coordYmax, coordXmin:coordXmax]
        #print(a)
    if OzoneEndFileValid:
        b = ozoneEnd[coordYmin:coordYmax, coordXmin:coordXmax]
        #print(b)
    ##
    # averaging the ozone values spatially and temporally...
    if OzoneStartFileValid and OzoneEndFileValid:
        auxData['ozone'] = np.mean((a+b)/2.)
    else:
        if OzoneStartFileValid and not OzoneEndFileValid:
            auxData['ozone'] = np.mean(a)
        elif not OzoneStartFileValid and OzoneEndFileValid:
            auxData['ozone'] = np.mean(b)
        #else: remain with default values.

    MetStartFileValid = os.path.getsize(AuxFullFilePath_dict['MET'][0]) > 1000
    MetEndFileValid = os.path.getsize(AuxFullFilePath_dict['MET'][1]) > 1000
    if MetStartFileValid:
        METStartProduct = snp.ProductIO.readProduct(AuxFullFilePath_dict['MET'][0])
    if MetEndFileValid:
        METEndProduct = snp.ProductIO.readProduct(AuxFullFilePath_dict['MET'][1])

    met_variable_names = {'pressure':'press', 'wind_u': 'z_wind', 'wind_v': 'm_wind', 'tcwv':'p_water'}

    for key in met_variable_names.keys():
        if MetStartFileValid:
            metStart = get_band_or_tiePointGrid(METStartProduct, met_variable_names[key], reshape=True)
            a = metStart[coordYmin:coordYmax, coordXmin:coordXmax]
        if MetEndFileValid:
            metEnd = get_band_or_tiePointGrid(METEndProduct,  met_variable_names[key], reshape=True)
            b = metEnd[coordYmin:coordYmax, coordXmin:coordXmax]
        ##
        # averaging the MET values spatially and temporally...
        if MetStartFileValid and MetEndFileValid:
            auxData[key] = np.mean((a + b) / 2.)
        else:
            if MetStartFileValid and not MetEndFileValid:
                auxData[key] = np.mean(a)
            elif not MetStartFileValid and MetEndFileValid:
                auxData[key] = np.mean(b)
            # else: remain with default values.

    if MetStartFileValid:
        METStartProduct.closeProductReader()
    if MetEndFileValid:
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

def check_downloadAuxDataFromArchive(year, doy, rep_path, type):
    url = "https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/"

    file_ext_list_Dict = {
        'ozone' : ["00_O3_AURAOMI_24h.hdf"],
        'met' :   ["00_MET_NCEPR2_6h.hdf.bz2", "00_MET_NCEP_6h.hdf", "06_MET_NCEPR2_6h.hdf.bz2",
                     "06_MET_NCEP_6h.hdf", "12_MET_NCEPR2_6h.hdf.bz2", "12_MET_NCEP_6h.hdf", "18_MET_NCEPR2_6h.hdf.bz2",
                     "18_MET_NCEP_6h.hdf"]
    }

    file_ext_list = file_ext_list_Dict[type]

    dest_path = os.path.join(rep_path, str(year))
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    dest_path = os.path.join(rep_path,str(year),'%03d'%int(doy))
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    for file_ext in file_ext_list:
        outname = "N%4d%03d" % (year, doy) + file_ext
        thisurl = url + "/N%4d%03d" % (year, doy) + file_ext

        if not os.path.exists(os.path.join(dest_path,outname)):
            try:
                print("Aux datafile missing, downloading %s"%thisurl)
                r = requests.get(thisurl, allow_redirects=True)
                # open method to open a file on your system and write the contents
                with open(os.path.join(dest_path,outname), "wb") as code:
                    code.write(r.content)
                code.close()
            except:
                print("Error in download, break")
                break
        else:
            #if 'O3' in file_ext:
            #    print(os.path.getsize(os.path.join(dest_path,outname)))
            if os.path.getsize(os.path.join(dest_path,outname)) < 1000:
                try:
                    print("Aux datafile too small, downloading %s"%thisurl)
                    r = requests.get(thisurl, allow_redirects=True)
                    # open method to open a file on your system and write the contents
                    with open(os.path.join(dest_path,outname), "wb") as code:
                        code.write(r.content)
                    code.close()
                except:
                    print("Error in download, break")
                    break


def checkAuxDataAvailablity(pathAuxDataRep, product=None, startTime=None, fmt='%Y%m%d'):
    if product is not None:
        startTime = product.getStartTime()
        dateMJD = startTime.getMJD()
    else:
        if startTime is None:
            print("provide a product or the starting time of the product.")
            sys.exit(1)
        else:
            dt = datetime.datetime.strptime(startTime, fmt)
            #tt = dt.timetuple()
            d0 = date(2000, 1, 1) #FIXME use the year given in startTime
            delta = dt.date() - d0
            print(delta.days)
            dateMJD = delta.days

    ###
    # for ozone:
    startTime, endTime = StartAndEndFileTimeMJD24H(dateMJD)
    startFilePrefix, endFilePrefix = InterpolationBorderComputer24H(dateMJD)

    if os.path.exists(pathAuxDataRep):
        year, doy, h = yearAndDoyAndHourUTC(startTime)
        startFilePath = os.path.join(pathAuxDataRep, str(year), "%03d"%doy)
        check_downloadAuxDataFromArchive(year, doy, pathAuxDataRep, type='ozone')
        files = recursive_glob(startFilePath,['.hdf','.hdf.bz2'])
        fullStartFilePathOzone = [fn for fn in files if os.path.join(startFilePath,startFilePrefix) in fn and 'O3' in fn]

        year, doy, h = yearAndDoyAndHourUTC(endTime)
        startFilePath = os.path.join(pathAuxDataRep, str(year), "%03d"%doy)
        check_downloadAuxDataFromArchive(year, doy, pathAuxDataRep, type='ozone')
        files = recursive_glob(startFilePath,['.hdf','.hdf.bz2'])
        fullEndFilePathOzone = [fn for fn in files if os.path.join(startFilePath,endFilePrefix) in fn and 'O3' in fn]

    ###
    # for meteorology:
    startTime, endTime = StartAndEndFileTimeMJD6H(dateMJD)
    startFilePrefix, endFilePrefix = InterpolationBorderComputer6H(dateMJD)

    if os.path.exists(pathAuxDataRep):

        year, doy, h = yearAndDoyAndHourUTC(startTime)
        startFilePath = os.path.join(pathAuxDataRep, str(year), "%03d"%doy)
        check_downloadAuxDataFromArchive(year, doy, pathAuxDataRep, type='met')
        files = recursive_glob(startFilePath,['.hdf','.hdf.bz2'])
        fullStartFilePathMET = [fn for fn in files if os.path.join(startFilePath,startFilePrefix) in fn and not '.bz2' in fn]

        year, doy, h = yearAndDoyAndHourUTC(endTime)
        startFilePath = os.path.join(pathAuxDataRep, str(year), "%03d"%doy)
        check_downloadAuxDataFromArchive(year, doy, pathAuxDataRep, type='met')
        files = recursive_glob(startFilePath,['.hdf','.hdf.bz2'])
        fullEndFilePathMET = [fn for fn in files if os.path.join(startFilePath,endFilePrefix) in fn and not '.bz2' in fn]

    out_dict = {
        'ozone': [fullStartFilePathOzone[0], fullEndFilePathOzone[0]],
        'MET': [fullStartFilePathMET[0], fullEndFilePathMET[0]]
    }

    return out_dict
